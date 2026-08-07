package gemmini

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters

import GemminiISA._

/** Fusion-only tags encoded in LocalAddr's otherwise-unused garbage field.
  * Keeping these helpers outside LocalAddr leaves the legacy address type
  * byte-for-byte identical to upstream Gemmini.
  */
object VpuLocalAddr {
  implicit class FusionOps(private val localAddr: LocalAddr) extends AnyVal {
    def a_from_vsram(dummy: Int = 0): Bool = flag(0, "A_FROM_VSRAM")
    def c_to_vsram(dummy: Int = 0): Bool = flag(1, "C_TO_VSRAM")
    def d_from_vsram(dummy: Int = 0): Bool = flag(2, "D_FROM_VSRAM")

    private def flag(bit: Int, name: String): Bool = {
      require(localAddr.garbage.getWidth > bit,
        s"$name requires LocalAddr garbage bit $bit")
      localAddr.garbage(bit)
    }
  }

  private def withGarbageFlag(
      localAddr: LocalAddr, bit: Int, value: Bool): LocalAddr = {
    val width = localAddr.garbage.getWidth
    require(width > bit,
      s"LocalAddr garbage field does not contain fusion flag bit $bit")

    val result = WireInit(localAddr)
    val mask = (BigInt(1) << bit).U(width.W)
    result.garbage := Mux(value, localAddr.garbage | mask,
      localAddr.garbage & ~mask)
    result
  }

  def with_a_from_vsram(localAddr: LocalAddr, enabled: Bool): LocalAddr =
    withGarbageFlag(localAddr, bit = 0, value = enabled)

  def with_c_to_vsram(localAddr: LocalAddr, enabled: Bool): LocalAddr =
    withGarbageFlag(localAddr, bit = 1, value = enabled)

  def with_d_from_vsram(localAddr: LocalAddr, enabled: Bool): LocalAddr =
    withGarbageFlag(localAddr, bit = 2, value = enabled)
}

import VpuLocalAddr._

/** Raw matrix-row transport shared by Gemmini and the standalone VPU.
  *
  * The bundle deliberately contains no Gemmini LocalAddr or VPU element
  * address.  Both accelerators agree that one row is `lanes` consecutive
  * elements; the fused configuration uses 16 FP32 elements per row.
  */
class GemminiVpuMatrixReadReq(rowAddrBits: Int) extends Bundle {
  val rowAddress = UInt(rowAddrBits.W)
}

class GemminiVpuMatrixReadResp(lanes: Int, elementBits: Int)
    extends Bundle {
  val data = Vec(lanes, UInt(elementBits.W))
}

class GemminiVpuMatrixWriteReq(
    rowAddrBits: Int,
    lanes: Int,
    elementBits: Int) extends Bundle {
  val rowAddress = UInt(rowAddrBits.W)
  val data = Vec(lanes, UInt(elementBits.W))
  val laneMask = Vec(lanes, Bool())
}

/** Gemmini-facing half of a matrix-read port.  Final accumulator writes are
  * emitted by SharedExtMem, because that is where the FP32 RMW result exists.
  */
class GemminiVpuMatrixReadIO(
    rowAddrBits: Int,
    lanes: Int,
    elementBits: Int) extends Bundle {
  val req = Decoupled(new GemminiVpuMatrixReadReq(rowAddrBits))
  val resp = Flipped(Decoupled(
    new GemminiVpuMatrixReadResp(lanes, elementBits)))
}

/** Shares one ordered VSRAM matrix-read port between the systolic A input and
  * the D-to-ACC mover.  The response has no transaction ID, so every accepted
  * request records its owner until the corresponding in-order response is
  * consumed. */
class GemminiVpuMatrixReadArbiter(
    rowAddrBits: Int,
    lanes: Int,
    elementBits: Int,
    ownerEntries: Int = 2) extends Module {
  require(ownerEntries >= 2)

  val io = IO(new Bundle {
    val execute = Flipped(new GemminiVpuMatrixReadIO(
      rowAddrBits, lanes, elementBits))
    val dload = Flipped(new GemminiVpuMatrixReadIO(
      rowAddrBits, lanes, elementBits))
    val out = new GemminiVpuMatrixReadIO(rowAddrBits, lanes, elementBits)
  })

  val requestArbiter = Module(new Arbiter(
    new GemminiVpuMatrixReadReq(rowAddrBits), 2))
  requestArbiter.io.in(0) <> io.execute.req
  requestArbiter.io.in(1) <> io.dload.req

  val owners = Module(new Queue(Bool(), ownerEntries, pipe = true))
  io.out.req.valid := requestArbiter.io.out.valid && owners.io.enq.ready
  io.out.req.bits := requestArbiter.io.out.bits
  requestArbiter.io.out.ready := io.out.req.ready && owners.io.enq.ready
  owners.io.enq.valid := io.out.req.fire
  owners.io.enq.bits := requestArbiter.io.chosen === 1.U

  val responseForDload = owners.io.deq.bits
  io.execute.resp.valid := io.out.resp.valid && owners.io.deq.valid &&
    !responseForDload
  io.execute.resp.bits := io.out.resp.bits
  io.dload.resp.valid := io.out.resp.valid && owners.io.deq.valid &&
    responseForDload
  io.dload.resp.bits := io.out.resp.bits
  io.out.resp.ready := owners.io.deq.valid && Mux(responseForDload,
    io.dload.resp.ready, io.execute.resp.ready)
  owners.io.deq.ready := io.out.resp.fire

  assert(!io.out.resp.valid || owners.io.deq.valid,
    "VSRAM matrix response arrived without an owner")
}

class VpuDLoadRowDescriptor(rowOffsetBits: Int) extends Bundle {
  val rowOffset = UInt(rowOffsetBits.W)
  val last = Bool()
}

class VpuDLoadWriteEntry(
    rowOffsetBits: Int,
    lanes: Int,
    elementBits: Int) extends Bundle {
  val rowOffset = UInt(rowOffsetBits.W)
  val last = Bool()
  val data = Vec(lanes, UInt(elementBits.W))
}

/** Executes fusion-tagged LOAD3 commands without issuing external memory
  * traffic. VSRAM requests are generated into a row queue and may continue
  * while earlier responses wait in an independent accumulator-write queue.
  * One FP32 matrix row is ultimately written verbatim into the corresponding
  * accumulator row. */
class VpuDLoadController[T <: Data : Arithmetic, U <: Data, V <: Data](
    config: GemminiArrayConfig[T, U, V])
    (implicit p: Parameters) extends Module {
  import config._

  private val blockRows = meshRows * tileRows
  private val blockCols = meshColumns * tileColumns
  private val accRowT = Vec(meshColumns, Vec(tileColumns, accType))
  private val vpuRowAddrBits =
    1 max log2Ceil(acc_banks * acc_bank_entries)
  private val vpuMatrixRows = acc_banks * acc_bank_entries
  require(acc_latency > 0)
  require(vpuRowAddrBits <= 32,
    "D_FROM_VSRAM encodes its matrix-row base in rs1[31:0]")
  private val rowOffsetBits = 1 max log2Ceil(blockRows)
  private val rowCountBits = 1 max log2Ceil(blockRows + 1)
  // Two entries per stage match the ordered matrix-read owner/response credit.
  // This is enough for an II=1 row pipeline without buffering a complete tile;
  // a longer ACC conflict propagates ordinary Decoupled backpressure upstream.
  private val readQueueEntries = 2
  private val outstandingQueueEntries = 2
  private val writeQueueEntries = 2

  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new GemminiCmd(
      reservation_station_entries)))
    val matrixRead = new GemminiVpuMatrixReadIO(
      vpuRowAddrBits, blockCols, accType.getWidth)
    val accWrite = Vec(acc_banks, Decoupled(new AccumulatorWriteReq(
      acc_bank_entries, accRowT)))
    val completed = Decoupled(UInt(
      log2Up(reservation_station_entries).W))
    val busy = Output(Bool())
  })

  object State extends ChiselEnum {
    val idle, transfer, complete = Value
  }
  import State._
  val state = RegInit(idle)

  val sourceBase = Reg(UInt(vpuRowAddrBits.W))
  val destinationBase = Reg(local_addr_t.cloneType)
  val rows = Reg(UInt((mvin_rows_bits max 1).W))
  val cols = Reg(UInt((mvin_cols_bits max 1).W))
  val robId = Reg(UInt(log2Up(reservation_station_entries).W))
  val generatedRows = RegInit(0.U(rowCountBits.W))
  val issuedRows = RegInit(0.U(rowCountBits.W))
  val respondedRows = RegInit(0.U(rowCountBits.W))
  val writtenRows = RegInit(0.U(rowCountBits.W))

  val readQueue = Module(new Queue(new VpuDLoadRowDescriptor(rowOffsetBits),
    readQueueEntries, pipe = true, flow = true))
  val outstandingQueue = Module(new Queue(
    new VpuDLoadRowDescriptor(rowOffsetBits),
    outstandingQueueEntries, pipe = true, flow = false))
  val writeQueue = Module(new Queue(new VpuDLoadWriteEntry(
    rowOffsetBits, blockCols, accType.getWidth),
    writeQueueEntries, pipe = true, flow = false))

  val decoded = io.cmd.bits.cmd.rs2.asTypeOf(new MvinRs2(
    mvin_rows_bits, mvin_cols_bits, local_addr_t))
  val transferActive = state === transfer

  // Expand one LOAD3 into ordered row requests. With flow enabled, a newly
  // generated row can reach the VSRAM port in the same cycle when it is ready.
  readQueue.io.enq.valid := transferActive && generatedRows < rows
  readQueue.io.enq.bits.rowOffset :=
    generatedRows(rowOffsetBits - 1, 0)
  readQueue.io.enq.bits.last := generatedRows === rows - 1.U
  when(readQueue.io.enq.fire) {
    generatedRows := generatedRows + 1.U
  }

  // A request may leave the row queue only when its response metadata can be
  // retained. These three handshakes therefore fire atomically.
  io.matrixRead.req.valid := transferActive && readQueue.io.deq.valid &&
    outstandingQueue.io.enq.ready
  io.matrixRead.req.bits.rowAddress := sourceBase +
    readQueue.io.deq.bits.rowOffset
  readQueue.io.deq.ready := transferActive && io.matrixRead.req.ready &&
    outstandingQueue.io.enq.ready
  outstandingQueue.io.enq.valid := transferActive &&
    readQueue.io.deq.valid && io.matrixRead.req.ready
  outstandingQueue.io.enq.bits := readQueue.io.deq.bits

  // Matrix responses carry no transaction ID. Pair each in-order response
  // with the oldest accepted request metadata, then retain the complete FP32
  // row independently until the destination ACC bank accepts it.
  writeQueue.io.enq.valid := transferActive && io.matrixRead.resp.valid &&
    outstandingQueue.io.deq.valid
  writeQueue.io.enq.bits.rowOffset :=
    outstandingQueue.io.deq.bits.rowOffset
  writeQueue.io.enq.bits.last := outstandingQueue.io.deq.bits.last
  writeQueue.io.enq.bits.data := io.matrixRead.resp.bits.data
  io.matrixRead.resp.ready := transferActive &&
    outstandingQueue.io.deq.valid && writeQueue.io.enq.ready
  outstandingQueue.io.deq.ready := transferActive &&
    io.matrixRead.resp.valid && writeQueue.io.enq.ready

  val readRequestFire = io.matrixRead.req.fire
  val readResponseFire = io.matrixRead.resp.fire
  when(readRequestFire) {
    assert(outstandingQueue.io.enq.fire,
      "D_FROM_VSRAM request lost its response metadata")
    issuedRows := issuedRows + 1.U
  }
  when(readResponseFire) {
    assert(outstandingQueue.io.deq.fire,
      "D_FROM_VSRAM response did not retire request metadata")
    assert(respondedRows < rows,
      "D_FROM_VSRAM returned more rows than requested")
    respondedRows := respondedRows + 1.U
  }
  when(io.matrixRead.resp.valid) {
    assert(outstandingQueue.io.deq.valid,
      "D_FROM_VSRAM response arrived without queued request metadata")
  }

  val currentDestination = destinationBase + writeQueue.io.deq.bits.rowOffset
  val selectedBank = currentDestination.acc_bank()

  io.cmd.ready := state === idle
  io.completed.valid := state === complete
  io.completed.bits := robId
  io.busy := state =/= idle
  dontTouch(io.busy)

  for (bank <- 0 until acc_banks) {
    val write = io.accWrite(bank)
    write.valid := transferActive && writeQueue.io.deq.valid &&
      selectedBank === bank.U
    write.bits.addr := currentDestination.acc_row()
    write.bits.data := writeQueue.io.deq.bits.data.asTypeOf(accRowT)
    write.bits.acc := false.B
    val bytesPerElement = accType.getWidth / 8
    require(bytesPerElement > 0)
    write.bits.mask := VecInit((0 until accRowT.getWidth / 8).map { byte =>
      (byte / bytesPerElement).U < cols
    })
  }

  val selectedWriteReady = if (acc_banks == 1) {
    io.accWrite.head.ready
  } else {
    Mux1H(UIntToOH(selectedBank, acc_banks), io.accWrite.map(_.ready))
  }
  writeQueue.io.deq.ready := transferActive && selectedWriteReady
  val writeFire = writeQueue.io.deq.fire
  when(writeFire) {
    writtenRows := writtenRows + 1.U
  }

  // These pulses make the intended overlap directly visible in waveforms.
  val dLoadReadIssue = readRequestFire
  val dLoadAccWriteIssue = writeFire
  val dLoadReadWriteOverlap = dLoadReadIssue && dLoadAccWriteIssue
  dontTouch(dLoadReadIssue)
  dontTouch(dLoadAccWriteIssue)
  dontTouch(dLoadReadWriteOverlap)

  when(transferActive) {
    assert(respondedRows <= issuedRows && issuedRows <= generatedRows &&
      generatedRows <= rows,
      "D_FROM_VSRAM row counters escaped their ordered bounds")
  }

  when (io.cmd.fire) {
    assert(io.cmd.bits.cmd.inst.funct === LOAD3_CMD,
      "D_FROM_VSRAM controller only accepts LOAD3")
    assert(decoded.local_addr.d_from_vsram(),
      "VSRAM D load is missing its LocalAddr tag")
    assert(decoded.local_addr.is_acc_addr && !decoded.local_addr.accumulate,
      "VSRAM D load must overwrite an accumulator row")
    assert(decoded.num_rows > 0.U && decoded.num_rows <= blockRows.U)
    assert(decoded.num_cols > 0.U && decoded.num_cols <= blockCols.U)
    assert(io.cmd.bits.cmd.rs1(31, 0) +& decoded.num_rows <=
      vpuMatrixRows.U,
      "D_FROM_VSRAM matrix-row range escaped VSRAM")
    sourceBase := io.cmd.bits.cmd.rs1(vpuRowAddrBits - 1, 0)
    destinationBase := decoded.local_addr
    rows := decoded.num_rows
    cols := decoded.num_cols
    robId := io.cmd.bits.rob_id.bits
    generatedRows := 0.U
    issuedRows := 0.U
    respondedRows := 0.U
    writtenRows := 0.U
    assert(!readQueue.io.deq.valid && !outstandingQueue.io.deq.valid &&
      !writeQueue.io.deq.valid,
      "D_FROM_VSRAM accepted a command with non-empty transfer queues")
    state := transfer
  }

  when (writeFire) {
    assert(writeQueue.io.deq.bits.rowOffset < rows,
      "D_FROM_VSRAM attempted to write an out-of-range row")
    assert(writtenRows < rows,
      "D_FROM_VSRAM wrote more rows than requested")
    when (writeQueue.io.deq.bits.last) {
      assert(writeQueue.io.deq.bits.rowOffset === rows - 1.U &&
        writtenRows === rows - 1.U,
        "D_FROM_VSRAM last-row metadata was out of order")
      assert(generatedRows === rows && issuedRows === rows &&
        respondedRows === rows && !readQueue.io.deq.valid &&
        !outstandingQueue.io.deq.valid,
        "D_FROM_VSRAM reached its final write with reads still pending")
      // Completion follows acceptance into the shared accumulator pipeline.
      // Subsequent commands use that same ordered pipeline, while the fixed
      // mesh-output latency separates a following execute write from this D
      // overwrite; the RS need not wait for the physical SRAM commit.
      state := complete
    }
  }

  when(state === complete) {
    assert(generatedRows === rows && issuedRows === rows &&
      respondedRows === rows && writtenRows === rows &&
      !readQueue.io.deq.valid && !outstandingQueue.io.deq.valid &&
      !writeQueue.io.deq.valid,
      "D_FROM_VSRAM completed before all queued rows drained")
  }

  when (io.completed.fire) {
    state := idle
  }
}
