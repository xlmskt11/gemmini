package gemmini

import scala.collection.mutable.ArrayBuffer

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.Decoupled
import freechips.rocketchip.diplomacy.{AddressSet, IdRange, LazyModule,
  LazyModuleImp}
import freechips.rocketchip.system.DefaultConfig
import freechips.rocketchip.tile.{RocketTileParams, TileKey,
  TileVisibilityNodeKey}
import freechips.rocketchip.tilelink.{TLClientNode, TLClientParameters,
  TLMasterPortParameters, TLRAM, TLEphemeralNode}
import org.chipsalliance.cde.config.Parameters

import GemminiISA._

private object LoopMatmulPagePackedTestConfig {
  val blockSize = 16
  val coreMaxAddrBits = 64
  val iteratorBitwidth = 16
  val maxAddr = 4096
  val inputWidth = 8
  val maxBlockLen = 8
  val concurrentLoops = 2
  val pagePackedFlag: BigInt = BigInt(1) << 31

  def mvinRs2: MvinRs2 = new MvinRs2(
    mvin_rows_bits = 5,
    mvin_cols_bits = 8,
    local_addr_t = new LocalAddr(
      sp_banks = 4,
      sp_bank_entries = 1024,
      acc_banks = 4,
      acc_bank_entries = 1024))
}

private case class PagePackedLoad(address: BigInt, rows: Int, cols: Int)
private case class PagePackedObservedLoad(
    address: BigInt, rows: Int, cols: Int, localAddr: Int)

private class PagePackedLoadCommand extends Bundle {
  val address = UInt(LoopMatmulPagePackedTestConfig.coreMaxAddrBits.W)
  val rs2 = UInt(64.W)
  val funct = UInt(7.W)
}

private class LoopMatmulPagePackedAHarness(implicit p: Parameters)
    extends LazyModule {
  import LoopMatmulPagePackedTestConfig._

  val visibilityNode = TLEphemeralNode()
  val dummyClient = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(
    TLClientParameters(
      name = "loop-matmul-page-packed-a-test",
      sourceId = IdRange(0, 1))))))
  val dummyRam = LazyModule(new TLRAM(
    AddressSet(0x0, 0xffff), beatBytes = 16))
  dummyRam.node := visibilityNode := dummyClient

  val dutParameters = p.alterMap(Map(
    TileKey -> RocketTileParams(),
    TileVisibilityNodeKey -> visibilityNode))

  lazy val module = new LoopMatmulPagePackedAHarnessImp(this)
}

private class LoopMatmulPagePackedAHarnessImp(
    outer: LoopMatmulPagePackedAHarness) extends LazyModuleImp(outer) {
  import LoopMatmulPagePackedTestConfig._

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new LoopMatmulLdAReq(
      blockSize, coreMaxAddrBits, iteratorBitwidth, maxAddr,
      concurrentLoops, use_shared_res_entries = true)))
    val cmd = Decoupled(new PagePackedLoadCommand)
    val idle = Output(Bool())
  })

  val dut = Module(new LoopMatmulLdA(
    block_size = blockSize,
    coreMaxAddrBits = coreMaxAddrBits,
    iterator_bitwidth = iteratorBitwidth,
    max_addr = maxAddr,
    input_w = inputWidth,
    max_block_len = maxBlockLen,
    concurrent_loops = concurrentLoops,
    mvin_rs2_t = mvinRs2,
    use_shared_res_entries = true)(outer.dutParameters))

  dut.io.req <> io.req
  io.cmd.valid := dut.io.cmd.valid
  dut.io.cmd.ready := io.cmd.ready
  io.cmd.bits.address := dut.io.cmd.bits.rs1
  io.cmd.bits.rs2 := dut.io.cmd.bits.rs2
  io.cmd.bits.funct := dut.io.cmd.bits.inst.funct
  io.idle := dut.io.idle
  dut.io.rob_overloaded := false.B
  dut.io.admission_blocked.get := false.B

  val (tl, _) = outer.dummyClient.out.head
  tl.a.valid := false.B
  tl.a.bits := DontCare
  tl.b.ready := true.B
  tl.c.valid := false.B
  tl.c.bits := DontCare
  tl.d.ready := true.B
  tl.e.valid := false.B
  tl.e.bits := DontCare
}

private class LoopMatmulPagePackedBHarness(implicit p: Parameters)
    extends LazyModule {
  import LoopMatmulPagePackedTestConfig._

  val visibilityNode = TLEphemeralNode()
  val dummyClient = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(
    TLClientParameters(
      name = "loop-matmul-page-packed-b-test",
      sourceId = IdRange(0, 1))))))
  val dummyRam = LazyModule(new TLRAM(
    AddressSet(0x0, 0xffff), beatBytes = 16))
  dummyRam.node := visibilityNode := dummyClient

  val dutParameters = p.alterMap(Map(
    TileKey -> RocketTileParams(),
    TileVisibilityNodeKey -> visibilityNode))

  lazy val module = new LoopMatmulPagePackedBHarnessImp(this)
}

private class LoopMatmulPagePackedBHarnessImp(
    outer: LoopMatmulPagePackedBHarness) extends LazyModuleImp(outer) {
  import LoopMatmulPagePackedTestConfig._

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new LoopMatmulLdBReq(
      blockSize, coreMaxAddrBits, iteratorBitwidth, maxAddr,
      concurrentLoops, use_shared_res_entries = true)))
    val cmd = Decoupled(new PagePackedLoadCommand)
    val idle = Output(Bool())
  })

  val dut = Module(new LoopMatmulLdB(
    block_size = blockSize,
    coreMaxAddrBits = coreMaxAddrBits,
    iterator_bitwidth = iteratorBitwidth,
    max_addr = maxAddr,
    input_w = inputWidth,
    max_block_len = maxBlockLen,
    concurrent_loops = concurrentLoops,
    mvin_rs2_t = mvinRs2,
    use_shared_res_entries = true,
    use_vpu_fusion = true)(outer.dutParameters))

  dut.io.req <> io.req
  io.cmd.valid := dut.io.cmd.valid
  dut.io.cmd.ready := io.cmd.ready
  io.cmd.bits.address := dut.io.cmd.bits.rs1
  io.cmd.bits.rs2 := dut.io.cmd.bits.rs2
  io.cmd.bits.funct := dut.io.cmd.bits.inst.funct
  io.idle := dut.io.idle
  dut.io.rob_overloaded := false.B
  dut.io.admission_blocked.get := false.B

  val (tl, _) = outer.dummyClient.out.head
  tl.a.valid := false.B
  tl.a.bits := DontCare
  tl.b.ready := true.B
  tl.c.valid := false.B
  tl.c.bits := DontCare
  tl.d.ready := true.B
  tl.e.valid := false.B
  tl.e.bits := DontCare
}

private trait PagePackedLoadCollector { this: PeekPokeTester[_] =>
  protected def decodeLoad(address: BigInt, rs2: BigInt): PagePackedLoad = {
    val cols = ((rs2 >> 32) & 0xffff).toInt
    val rows = ((rs2 >> 48) & 0xffff).toInt
    PagePackedLoad(address, rows, cols)
  }

  protected def decodeObservedLoad(
      address: BigInt, rs2: BigInt): PagePackedObservedLoad = {
    val decoded = decodeLoad(address, rs2)
    val localAddr = (rs2 &
      (LoopMatmulPagePackedTestConfig.maxAddr - 1)).toInt
    PagePackedObservedLoad(decoded.address, decoded.rows, decoded.cols,
      localAddr)
  }
}

private class LoopMatmulPagePackedATester(c: LoopMatmulPagePackedAHarnessImp)
    extends PeekPokeTester(c) with PagePackedLoadCollector {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_i, 4)
  poke(c.io.req.bits.max_k, 2)
  poke(c.io.req.bits.global_i.get, 32)
  poke(c.io.req.bits.global_k.get, 2)
  poke(c.io.req.bits.i_offset.get, 1)
  poke(c.io.req.bits.k_offset.get, 0)
  poke(c.io.req.bits.pad_i, 0)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.dram_addr, 0x10000)
  poke(c.io.req.bits.dram_stride, pagePackedFlag | 256)
  // LOOP_WS offsets are logical (I, K). A transpose maps them to physical
  // (row, col) = (K, I), and the shared I shard offset also belongs to col.
  poke(c.io.req.bits.tile_row_offset, 6)
  poke(c.io.req.bits.tile_col_offset, 3)
  poke(c.io.req.bits.transpose, true)
  poke(c.io.req.bits.addr_start, 0)
  poke(c.io.req.bits.loop_id, 0)

  expect(c.io.req.ready, true)
  poke(c.io.req.valid, true)
  step(1)
  poke(c.io.req.valid, false)

  val loads = ArrayBuffer.empty[PagePackedLoad]
  var cycles = 0
  while (loads.size < 4 && cycles < 20) {
    if (peek(c.io.cmd.valid) != 0 && peek(c.io.cmd.ready) != 0) {
      loads += decodeLoad(peek(c.io.cmd.bits.address), peek(c.io.cmd.bits.rs2))
      expect(c.io.cmd.bits.funct, LOAD_CMD)
    }
    step(1)
    cycles += 1
  }

  assert(loads.size == 4, s"timed out after collecting ${loads.size} A loads")
  assert(loads.toSeq == Seq(
    PagePackedLoad(0x12870, 16, 16),
    PagePackedLoad(0x13800, 16, 48),
    PagePackedLoad(0x14070, 16, 16),
    PagePackedLoad(0x15000, 16, 48)),
    s"unexpected transpose-aware page-packed A loads: $loads")
  expect(c.io.idle, true)
}

private class LoopMatmulPagePackedBTester(c: LoopMatmulPagePackedBHarnessImp)
    extends PeekPokeTester(c) with PagePackedLoadCollector {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_k, 4)
  poke(c.io.req.bits.max_j, 2)
  poke(c.io.req.bits.global_k.get, 32)
  poke(c.io.req.bits.global_j.get, 2)
  poke(c.io.req.bits.k_offset.get, 1)
  poke(c.io.req.bits.j_offset.get, 0)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.pad_j, 0)
  poke(c.io.req.bits.dram_addr, 0x20000)
  poke(c.io.req.bits.dram_stride, pagePackedFlag | 512)
  // LOOP_WS offsets are logical (K, J). B transpose maps them to physical
  // (row, col) = (J, K), and the shared K shard offset also belongs to col.
  poke(c.io.req.bits.tile_row_offset, 14)
  poke(c.io.req.bits.tile_col_offset, 2)
  poke(c.io.req.bits.transpose, true)
  poke(c.io.req.bits.addr_end, maxAddr)
  poke(c.io.req.bits.loop_id, 0)

  expect(c.io.req.ready, true)
  poke(c.io.req.valid, true)
  step(1)
  poke(c.io.req.valid, false)

  val loads = ArrayBuffer.empty[PagePackedLoad]
  var cycles = 0
  while (loads.size < 4 && cycles < 20) {
    if (peek(c.io.cmd.valid) != 0 && peek(c.io.cmd.ready) != 0) {
      loads += decodeLoad(peek(c.io.cmd.bits.address), peek(c.io.cmd.bits.rs2))
      expect(c.io.cmd.bits.funct, LOAD2_CMD)
    }
    step(1)
    cycles += 1
  }

  assert(loads.size == 4, s"timed out after collecting ${loads.size} B loads")
  assert(loads.toSeq == Seq(
    PagePackedLoad(0x240f0, 16, 16),
    PagePackedLoad(0x260f0, 16, 16),
    PagePackedLoad(0x25000, 16, 48),
    PagePackedLoad(0x27000, 16, 48)),
    s"unexpected transpose-aware page-packed B loads: $loads")
  expect(c.io.idle, true)
}

private class LoopMatmulLdBZeroJTester(c: LoopMatmulPagePackedBHarnessImp)
    extends PeekPokeTester(c) {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_k, 4)
  poke(c.io.req.bits.max_j, 0)
  poke(c.io.req.bits.global_k.get, 4)
  poke(c.io.req.bits.global_j.get, 8)
  poke(c.io.req.bits.k_offset.get, 0)
  poke(c.io.req.bits.j_offset.get, 8)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.pad_j, 0)
  poke(c.io.req.bits.dram_addr, 0x20000)
  poke(c.io.req.bits.dram_stride, pagePackedFlag | 512)
  poke(c.io.req.bits.tile_row_offset, 0)
  poke(c.io.req.bits.tile_col_offset, 0)
  poke(c.io.req.bits.transpose, false)
  poke(c.io.req.bits.addr_end, maxAddr)
  poke(c.io.req.bits.loop_id, 0)

  poke(c.io.req.valid, true)
  step(1)
  poke(c.io.req.valid, false)

  // A zero-column local B stream (for example, an empty N partition) remains
  // a valid group participant but must close without an underflowed LOAD2.
  expect(c.io.cmd.valid, false)
  step(1)
  expect(c.io.cmd.valid, false)
  expect(c.io.idle, true)
}

private class LoopMatmulPagePackedAOffsetTester(
    c: LoopMatmulPagePackedAHarnessImp,
    maxI: Int,
    maxK: Int,
    globalI: Int,
    globalK: Int,
    iOffset: Int,
    kOffset: Int,
    dramBase: BigInt,
    tileRowOffset: Int,
    tileColOffset: Int,
    transpose: Boolean,
    expected: Seq[PagePackedObservedLoad])
    extends PeekPokeTester(c) with PagePackedLoadCollector {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_i, maxI)
  poke(c.io.req.bits.max_k, maxK)
  poke(c.io.req.bits.global_i.get, globalI)
  poke(c.io.req.bits.global_k.get, globalK)
  poke(c.io.req.bits.i_offset.get, iOffset)
  poke(c.io.req.bits.k_offset.get, kOffset)
  poke(c.io.req.bits.pad_i, 0)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.dram_addr, dramBase)
  poke(c.io.req.bits.dram_stride, pagePackedFlag | 256)
  poke(c.io.req.bits.tile_row_offset, tileRowOffset)
  poke(c.io.req.bits.tile_col_offset, tileColOffset)
  poke(c.io.req.bits.transpose, transpose)
  poke(c.io.req.bits.addr_start, 0)
  poke(c.io.req.bits.loop_id, 0)

  expect(c.io.req.ready, true)
  poke(c.io.req.valid, true)
  step(1)
  poke(c.io.req.valid, false)

  val loads = ArrayBuffer.empty[PagePackedObservedLoad]
  var cycles = 0
  while (loads.size < expected.size && cycles < 20) {
    if (peek(c.io.cmd.valid) != 0 && peek(c.io.cmd.ready) != 0) {
      loads += decodeObservedLoad(peek(c.io.cmd.bits.address),
        peek(c.io.cmd.bits.rs2))
      expect(c.io.cmd.bits.funct, LOAD_CMD)
    }
    step(1)
    cycles += 1
  }

  assert(loads.toSeq == expected,
    s"unexpected page-packed A offset/stride loads: $loads")
  expect(c.io.idle, true)
}

private class LoopMatmulPagePackedBOffsetTester(
    c: LoopMatmulPagePackedBHarnessImp,
    maxK: Int,
    maxJ: Int,
    globalK: Int,
    globalJ: Int,
    kOffset: Int,
    jOffset: Int,
    dramBase: BigInt,
    tileRowOffset: Int,
    tileColOffset: Int,
    transpose: Boolean,
    expected: Seq[PagePackedObservedLoad])
    extends PeekPokeTester(c) with PagePackedLoadCollector {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_k, maxK)
  poke(c.io.req.bits.max_j, maxJ)
  poke(c.io.req.bits.global_k.get, globalK)
  poke(c.io.req.bits.global_j.get, globalJ)
  poke(c.io.req.bits.k_offset.get, kOffset)
  poke(c.io.req.bits.j_offset.get, jOffset)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.pad_j, 0)
  poke(c.io.req.bits.dram_addr, dramBase)
  poke(c.io.req.bits.dram_stride, pagePackedFlag | 512)
  poke(c.io.req.bits.tile_row_offset, tileRowOffset)
  poke(c.io.req.bits.tile_col_offset, tileColOffset)
  poke(c.io.req.bits.transpose, transpose)
  poke(c.io.req.bits.addr_end, maxAddr)
  poke(c.io.req.bits.loop_id, 0)

  expect(c.io.req.ready, true)
  poke(c.io.req.valid, true)
  step(1)
  poke(c.io.req.valid, false)

  val loads = ArrayBuffer.empty[PagePackedObservedLoad]
  var cycles = 0
  while (loads.size < expected.size && cycles < 20) {
    if (peek(c.io.cmd.valid) != 0 && peek(c.io.cmd.ready) != 0) {
      loads += decodeObservedLoad(peek(c.io.cmd.bits.address),
        peek(c.io.cmd.bits.rs2))
      expect(c.io.cmd.bits.funct, LOAD2_CMD)
    }
    step(1)
    cycles += 1
  }

  assert(loads.toSeq == expected,
    s"unexpected page-packed B offset/stride loads: $loads")
  expect(c.io.idle, true)
}

class LoopMatmulPagePackedUnitTest extends ChiselFlatSpec {
  import LoopMatmulPagePackedTestConfig._

  implicit val p: Parameters = (new DefaultConfig).toInstance

  behavior of "LoopMatmul page-packed loads"

  it should "generate transpose-aware page-packed A addresses" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-a")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedAHarness).module) { c =>
      new LoopMatmulPagePackedATester(c)
    } should be(true)
  }

  it should "generate transpose-aware page-packed B addresses" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-b")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedBHarness).module) { c =>
      new LoopMatmulPagePackedBTester(c)
    } should be(true)
  }

  it should "close a zero-column B stream without emitting a LOAD2 command" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-ldb-zero-j")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedBHarness).module) { c =>
      new LoopMatmulLdBZeroJTester(c)
    } should be(true)
  }

  it should "apply the N-split auxiliary I offset to transposed A" in {
    val expected = Seq(
      PagePackedObservedLoad(0x34870, 16, 16, 0x030),
      PagePackedObservedLoad(0x35800, 16, 16, 0x040),
      PagePackedObservedLoad(0x36070, 16, 16, 0x0b0),
      PagePackedObservedLoad(0x37000, 16, 16, 0x0c0))
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-n-a")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedAHarness).module) { c =>
      new LoopMatmulPagePackedAOffsetTester(c,
        maxI = 2, maxK = 2, globalI = 8, globalK = 2,
        iOffset = 3, kOffset = 0, dramBase = 0x30000,
        tileRowOffset = 4, tileColOffset = 5, transpose = true,
        expected = expected)
    } should be(true)
  }

  it should "use the N-split local J bound and global-J B stride" in {
    val expected = Seq(
      PagePackedObservedLoad(0x44070, 16, 32, 0xf30),
      PagePackedObservedLoad(0x46070, 16, 32, 0xfb0))
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-n-b")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedBHarness).module) { c =>
      new LoopMatmulPagePackedBOffsetTester(c,
        maxK = 2, maxJ = 2, globalK = 2, globalJ = 8,
        kOffset = 0, jOffset = 3, dramBase = 0x40000,
        tileRowOffset = 2, tileColOffset = 4, transpose = false,
        expected = expected)
    } should be(true)
  }

  it should "apply the K-split local/global K fields to A" in {
    val expected = Seq(
      PagePackedObservedLoad(0x54870, 16, 16, 0x030),
      PagePackedObservedLoad(0x56070, 16, 16, 0x0b0),
      PagePackedObservedLoad(0x55800, 16, 16, 0x040),
      PagePackedObservedLoad(0x57000, 16, 16, 0x0c0))
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-k-a")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedAHarness).module) { c =>
      new LoopMatmulPagePackedAOffsetTester(c,
        maxI = 2, maxK = 2, globalI = 2, globalK = 8,
        iOffset = 0, kOffset = 3, dramBase = 0x50000,
        tileRowOffset = 5, tileColOffset = 4, transpose = false,
        expected = expected)
    } should be(true)
  }

  it should "apply the K-split global-K stride to transposed B" in {
    val expected = Seq(
      PagePackedObservedLoad(0x64090, 16, 32, 0xf30),
      PagePackedObservedLoad(0x66090, 16, 32, 0xfb0))
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/loop-matmul-page-packed-k-b")

    chisel3.iotesters.Driver.execute(args,
      () => LazyModule(new LoopMatmulPagePackedBHarness).module) { c =>
      new LoopMatmulPagePackedBOffsetTester(c,
        maxK = 2, maxJ = 2, globalK = 8, globalJ = 2,
        kOffset = 3, jOffset = 0, dramBase = 0x60000,
        tileRowOffset = 6, tileColOffset = 2, transpose = true,
        expected = expected)
    } should be(true)
  }

}
