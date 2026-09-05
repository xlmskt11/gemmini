package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.{log2Ceil, PopCount}

import Arithmetic.FloatArithmetic._

private object PrivateAccumulatorEndpointTestConfig {
  val value = GemminiFPConfigs.defaultFPConfig.copy(
    inputType = Float(8, 24), spatialArrayOutputType = Float(8, 24),
    accType = Float(8, 24), tileRows = 1, tileColumns = 1,
    meshRows = 8, meshColumns = 8, dataflow = Dataflow.WS,
    sp_banks = 4, sp_sub_banks = 1, sp_capacity = CapacityInMatrices(64),
    acc_banks = 4, acc_sub_banks = 2,
    acc_capacity = CapacityInMatrices(32), acc_latency = 4,
    use_shared_ext_mem = false, use_shared_res_entries = false,
    use_vpu_fusion = true, nSharers = 4,
    has_training_convs = false, has_max_pool = false,
    has_nonlinear_activations = false)
}

/** One private 8x8 endpoint with four logical ACC banks and two physical
  * sub-banks apiece. Bank client 0 is Gemmini and client 1 is the atomic VPU
  * adapter. */
private class PrivateAccumulatorEndpointHarness extends Module {
  private val config = PrivateAccumulatorEndpointTestConfig.value
  import config._
  private val lanes = 16
  private val tagBits = 4
  private val blockCols = meshColumns * tileColumns
  private val rowType = Vec(meshColumns, Vec(tileColumns, accType))
  private val depth = acc_bank_entries
  private val physicalDepth = depth / acc_sub_banks
  private val elementAddrBits = 1 max log2Ceil(
    acc_banks * acc_bank_entries * blockCols)

  val io = IO(new Bundle {
    val vpu = Flipped(new GemminiVpuMemoryIO(
      elementAddrBits, lanes, accType.getWidth, tagBits))
    val gemmini = Vec(acc_banks, Flipped(new ExtAccumulatorBankIO(
      depth, rowType, acc_banks)))
    val vpuPhysicalReadFire = Output(
      Vec(acc_banks, Vec(acc_sub_banks, Bool())))
    val vpuPhysicalWriteFire = Output(
      Vec(acc_banks, Vec(acc_sub_banks, Bool())))
  })

  private val adapter = Module(new VpuSharedAccumulatorAdapter(
    config, vpuLanes = lanes, tagBits = tagBits))
  private val gemminiAdapters = Seq.fill(acc_banks) {
    Module(new ExtAccSubBankAdapter(
      depth, rowType, acc_scale_t, acc_banks, acc_sub_banks,
      enableExRead = true, swizzleShift = log2Ceil(meshRows * tileRows)))
  }
  private val banks = Seq.tabulate(acc_banks, acc_sub_banks) { (_, _) =>
    Module(new ExtAccBank(
      nSharers = 2, n = physicalDepth, t = rowType, accBanks = acc_banks,
      acc_singleported = false, acc_latency = acc_latency,
      enableExWrite = false, atomicClient = 1))
  }

  for (client <- io.vpu.readRequest.indices) {
    adapter.io.vpu.readRequest(client).valid := io.vpu.readRequest(client).valid
    adapter.io.vpu.readRequest(client).bits := io.vpu.readRequest(client).bits
    io.vpu.readRequest(client).ready := adapter.io.vpu.readRequest(client).ready
    io.vpu.readResponse(client).valid := adapter.io.vpu.readResponse(client).valid
    io.vpu.readResponse(client).bits := adapter.io.vpu.readResponse(client).bits
    adapter.io.vpu.readResponse(client).ready := io.vpu.readResponse(client).ready
  }
  for (client <- io.vpu.writeRequest.indices) {
    adapter.io.vpu.writeRequest(client).valid := io.vpu.writeRequest(client).valid
    adapter.io.vpu.writeRequest(client).bits := io.vpu.writeRequest(client).bits
    io.vpu.writeRequest(client).ready := adapter.io.vpu.writeRequest(client).ready
  }
  io.vpu.busy := adapter.io.vpu.busy
  io.vpu.readConflictStall := adapter.io.vpu.readConflictStall
  io.vpu.writeConflictStall := adapter.io.vpu.writeConflictStall

  for (bank <- 0 until acc_banks) {
    val logical = gemminiAdapters(bank).io.bank
    val gemmini = io.gemmini(bank)
    logical.read.req.valid := gemmini.read.req.valid
    logical.read.req.bits.addr := gemmini.read.req.bits.addr
    logical.read.req.bits.full := gemmini.read.req.bits.full
    logical.read.req.bits.fromDMA := false.B
    logical.read.req.bits.scale := 0.U.asTypeOf(acc_scale_t)
    logical.read.req.bits.igelu_qb := 0.U.asTypeOf(accType)
    logical.read.req.bits.igelu_qc := 0.U.asTypeOf(accType)
    logical.read.req.bits.iexp_qln2 := 0.U.asTypeOf(accType)
    logical.read.req.bits.iexp_qln2_inv := 0.U.asTypeOf(accType)
    logical.read.req.bits.act := 0.U
    gemmini.read.req.ready := logical.read.req.ready
    gemmini.read.resp.valid := logical.read.resp.valid
    gemmini.read.resp.bits.data := logical.read.resp.bits.data
    gemmini.read.resp.bits.acc_bank_id := bank.U
    logical.read.resp.ready := gemmini.read.resp.ready
    logical.write.valid := gemmini.write.valid
    logical.write.bits := gemmini.write.bits
    gemmini.write.ready := logical.write.ready
    logical.grant.valid := false.B
    logical.grant.bits := DontCare
    gemmini.grant.ready := true.B
    for (subBank <- 0 until acc_sub_banks) {
      gemminiAdapters(bank).io.ext(subBank) <> banks(bank)(subBank).io.in(0)
      adapter.io.memory(bank)(subBank) <> banks(bank)(subBank).io.in(1)
      adapter.io.atomic(bank)(subBank) <> banks(bank)(subBank).io.atomic.get
      io.vpuPhysicalReadFire(bank)(subBank) :=
        adapter.io.memory(bank)(subBank).read.req.fire
      io.vpuPhysicalWriteFire(bank)(subBank) :=
        adapter.io.memory(bank)(subBank).write.fire
    }
  }

  private val flatBanks = banks.flatten
  private val adder = Module(new AccPipeShared(
    acc_latency - 1, rowType, flatBanks.size))
  private val adderValids = VecInit(flatBanks.map(_.io.adder.valid))
  assert(PopCount(adderValids) <= 1.U)
  for ((bank, index) <- flatBanks.zipWithIndex) {
    adder.io.in_sel(index) := bank.io.adder.valid
    adder.io.ina(index) := bank.io.adder.op1
    adder.io.inb(index) := bank.io.adder.op2
    bank.io.adder.sum := adder.io.out
  }
}

private class PrivateAccumulatorEndpointTester(
    c: PrivateAccumulatorEndpointHarness) extends PeekPokeTester(c) {
  private val accLatency = 4

  private def vpuRead(client: Int, valid: Boolean, tag: Int = 0): Unit = {
    poke(c.io.vpu.readRequest(client).valid, valid)
    poke(c.io.vpu.readRequest(client).bits.address, 0)
    poke(c.io.vpu.readRequest(client).bits.fragmentStride, 8)
    poke(c.io.vpu.readRequest(client).bits.tag, tag)
    c.io.vpu.readRequest(client).bits.laneMask.foreach(poke(_, valid))
  }
  private def vpuWrite(valid: Boolean, data: Seq[BigInt]): Unit = {
    poke(c.io.vpu.writeRequest(0).valid, valid)
    poke(c.io.vpu.writeRequest(0).bits.address, 0)
    poke(c.io.vpu.writeRequest(0).bits.fragmentStride, 8)
    for (lane <- 0 until 16) {
      poke(c.io.vpu.writeRequest(0).bits.laneMask(lane), valid)
      poke(c.io.vpu.writeRequest(0).bits.data(lane), data(lane))
    }
  }
  private def gemRead(bank: Int, valid: Boolean, row: Int = 0): Unit = {
    poke(c.io.gemmini(bank).read.req.valid, valid)
    poke(c.io.gemmini(bank).read.req.bits.addr, row)
    poke(c.io.gemmini(bank).read.req.bits.full, true)
  }
  private def gemWrite(bank: Int, row: Int, data: Seq[BigInt]): Unit = {
    val port = c.io.gemmini(bank).write
    poke(port.valid, true); poke(port.bits.addr, row)
    poke(port.bits.acc, false); poke(port.bits.exwrite, false)
    port.bits.data.flatten.zip(data).foreach { case (e, v) => poke(e.bits, v) }
    port.bits.mask.foreach(poke(_, true))
    expect(port.ready, true); step(1); poke(port.valid, false)
  }
  private def expectGem(bank: Int, data: Seq[BigInt]): Unit = {
    expect(c.io.gemmini(bank).read.resp.valid, true)
    c.io.gemmini(bank).read.resp.bits.data.flatten.zip(data).foreach {
      case (e, v) => expect(e.bits, v)
    }
  }
  private def expectVpu(client: Int, tag: Int, data: Seq[BigInt]): Unit = {
    expect(c.io.vpu.readResponse(client).valid, true)
    expect(c.io.vpu.readResponse(client).bits.tag, tag)
    c.io.vpu.readResponse(client).bits.data.zip(data).foreach {
      case (e, v) => expect(e, v)
    }
  }

  for (client <- 0 until 3) {
    vpuRead(client, valid = false); poke(c.io.vpu.readResponse(client).ready, false)
  }
  for (client <- 0 until 2) {
    poke(c.io.vpu.writeRequest(client).valid, false)
    poke(c.io.vpu.writeRequest(client).bits.address, 0)
    poke(c.io.vpu.writeRequest(client).bits.fragmentStride, 8)
    c.io.vpu.writeRequest(client).bits.laneMask.foreach(poke(_, false))
    c.io.vpu.writeRequest(client).bits.data.foreach(poke(_, 0))
  }
  for (bank <- 0 until 4) {
    gemRead(bank, valid = false); poke(c.io.gemmini(bank).read.resp.ready, true)
    poke(c.io.gemmini(bank).write.valid, false)
    poke(c.io.gemmini(bank).grant.valid, false); poke(c.io.gemmini(bank).grant.bits, false)
  }
  step(2)

  val row0 = (0 until 8).map(i => BigInt(0x1000 + i))
  val row1 = (0 until 8).map(i => BigInt(0x1100 + i))
  val row2 = (0 until 8).map(i => BigInt(0x1200 + i))
  val bank1Row = (0 until 8).map(i => BigInt(0x1300 + i))
  gemWrite(0, 0, row0); gemWrite(0, 1, row1); gemWrite(0, 2, row2)
  gemWrite(1, 0, bank1Row); step(accLatency + 1)

  // An uncontended DIM8 read accepts both adjacent physical sub-banks in the
  // same cycle. It also leaves both bank arbiters pointing back at Gemmini so
  // the following contention case is deterministic.
  vpuRead(0, valid = true, tag = 8)
  expect(c.io.vpuPhysicalReadFire(0)(0), true)
  expect(c.io.vpuPhysicalReadFire(0)(1), true)
  expect(c.io.vpu.readRequest(0).ready, true); step(1)
  vpuRead(0, valid = false); expectVpu(0, 8, row0 ++ row1)
  poke(c.io.vpu.readResponse(0).ready, true); step(1)
  poke(c.io.vpu.readResponse(0).ready, false)

  // Gemmini first wins sub-bank 0. The VPU waits rather than partially
  // issuing, then fires both adjacent DIM8 rows atomically on sub-banks 0/1.
  vpuRead(0, valid = true, tag = 9); gemRead(0, valid = true, row = 2)
  expect(c.io.vpuPhysicalReadFire(0)(0), false)
  expect(c.io.vpuPhysicalReadFire(0)(1), false)
  expect(c.io.vpu.readRequest(0).ready, false)
  expect(c.io.gemmini(0).read.req.ready, true); step(1)
  gemRead(0, valid = false); expectGem(0, row2)
  expect(c.io.vpuPhysicalReadFire(0)(0), true)
  expect(c.io.vpuPhysicalReadFire(0)(1), true)
  expect(c.io.vpu.readRequest(0).ready, true); step(1)
  vpuRead(0, valid = false); expectVpu(0, 9, row0 ++ row1)
  poke(c.io.vpu.readResponse(0).ready, true); step(1)
  poke(c.io.vpu.readResponse(0).ready, false)

  // Bank 1 Gemmini and bank 0 VPU fire concurrently.
  vpuRead(1, valid = true, tag = 6); gemRead(1, valid = true)
  expect(c.io.vpuPhysicalReadFire(0)(0), true)
  expect(c.io.vpuPhysicalReadFire(0)(1), true)
  expect(c.io.vpuPhysicalReadFire(1)(0), false)
  expect(c.io.vpuPhysicalReadFire(1)(1), false)
  expect(c.io.gemmini(1).read.req.ready, true)
  expect(c.io.vpu.readRequest(1).ready, true); step(1)
  vpuRead(1, valid = false); expectVpu(1, 6, row0 ++ row1)
  gemRead(1, valid = false); expectGem(1, bank1Row)
  poke(c.io.vpu.readResponse(1).ready, true); step(1)
  poke(c.io.vpu.readResponse(1).ready, false)

  // The VPU write also fires both physical sub-banks in one cycle. Each
  // pending row blocks a same-address Gemmini read until the physical ACC
  // pipeline commits, after which the new data is visible.
  val newData = (0 until 16).map(i => BigInt(0x2000 + i))
  vpuWrite(valid = true, newData)
  expect(c.io.vpuPhysicalWriteFire(0)(0), true)
  expect(c.io.vpuPhysicalWriteFire(0)(1), true)
  expect(c.io.vpu.writeRequest(0).ready, true); step(1)
  vpuWrite(valid = false, newData)
  gemRead(0, valid = true)
  expect(c.io.gemmini(0).read.req.ready, false); step(1)
  for (_ <- 1 until accLatency) {
    expect(c.io.gemmini(0).read.req.ready, false); step(1)
  }
  expect(c.io.gemmini(0).read.req.ready, true); step(1)
  gemRead(0, valid = false); expectGem(0, newData.take(8))
}

class PrivateAccumulatorEndpointUnitTest extends ChiselFlatSpec {
  behavior of "a private Gemmini/VPU accumulator endpoint"
  it should "arbitrate real banks and preserve DIM8 data, tags, and hazards" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/private-acc-endpoint")
    chisel3.iotesters.Driver.execute(args, () =>
      new PrivateAccumulatorEndpointHarness) { c =>
      new PrivateAccumulatorEndpointTester(c)
    } should be(true)
  }
}
