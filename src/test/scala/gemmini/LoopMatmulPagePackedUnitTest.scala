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
  val nSharers = 4
  val groupWidth = 3

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
      concurrentLoops, groupWidth, nSharers,
      use_shared_res_entries = true, use_vpu_fusion = true)))
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
    use_vpu_fusion = true,
    group_w = groupWidth,
    nSharers = nSharers)(outer.dutParameters))

  dut.io.req <> io.req
  io.cmd.valid := dut.io.cmd.valid
  dut.io.cmd.ready := io.cmd.ready
  io.cmd.bits.address := dut.io.cmd.bits.rs1
  io.cmd.bits.rs2 := dut.io.cmd.bits.rs2
  io.cmd.bits.funct := dut.io.cmd.bits.inst.funct
  io.idle := dut.io.idle
  dut.io.rob_overloaded := false.B
  dut.io.loop_full.get := false.B

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
}

private class LoopMatmulPagePackedATester(c: LoopMatmulPagePackedAHarnessImp)
    extends PeekPokeTester(c) with PagePackedLoadCollector {
  import LoopMatmulPagePackedTestConfig._

  reset(5)
  poke(c.io.cmd.ready, true)
  poke(c.io.req.valid, false)

  poke(c.io.req.bits.max_i, 4)
  poke(c.io.req.bits.max_k, 2)
  poke(c.io.req.bits.max_fi.get, 32)
  poke(c.io.req.bits.pad_i, 0)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.laddrR_offset.get, 1)
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
  poke(c.io.req.bits.max_fk.get, 32)
  poke(c.io.req.bits.max_j, 2)
  poke(c.io.req.bits.pad_k, 0)
  poke(c.io.req.bits.pad_j, 0)
  poke(c.io.req.bits.laddrR_offset.get, 1)
  poke(c.io.req.bits.group_id.get, 0)
  poke(c.io.req.bits.group_list.get, 0xf)
  poke(c.io.req.bits.has_gemv_followup.get, false)
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
}
