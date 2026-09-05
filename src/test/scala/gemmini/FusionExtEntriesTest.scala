package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.{Cat, UIntToOH}
import Util._

class FusionExtEntriesHarness extends Module {
  private val entriesPerQueue = 2
  private val localAddrType = new LocalAddr(
    sp_banks = 4, sp_bank_entries = 64,
    acc_banks = 4, acc_bank_entries = 64)

  val io = IO(new Bundle {
    val gemAllocValid = Input(Bool())
    val gemAllocQueue = Input(UInt(2.W))
    val gemAllocSlot = Input(UInt(1.W))
    val gemOpaStart = Input(UInt(8.W))
    val gemOpaEnd = Input(UInt(8.W))
    val gemOpaIsDst = Input(Bool())
    val gemOpbValid = Input(Bool())
    val gemOpbStart = Input(UInt(8.W))
    val gemOpbEnd = Input(UInt(8.W))
    val gemOpcValid = Input(Bool())
    val gemOpcStart = Input(UInt(8.W))
    val gemOpcEnd = Input(UInt(8.W))

    val gemIssueValid = Input(Bool())
    val gemIssueQueue = Input(UInt(2.W))
    val gemIssueSlot = Input(UInt(1.W))
    val gemIssueKeep = Input(Bool())
    val gemCompleteValid = Input(Bool())
    val gemCompleteQueue = Input(UInt(2.W))
    val gemCompleteSlot = Input(UInt(1.W))

    val vpuAllocValid = Input(Bool())
    val vpuAllocQueue = Input(UInt(2.W))
    val vpuAllocSlot = Input(UInt(1.W))
    val vpuOpaStart = Input(UInt(8.W))
    val vpuOpaEnd = Input(UInt(8.W))
    val vpuOpaIsDst = Input(Bool())
    val vpuOpbValid = Input(Bool())
    val vpuOpbStart = Input(UInt(8.W))
    val vpuOpbEnd = Input(UInt(8.W))
    val vpuOpcValid = Input(Bool())
    val vpuOpcStart = Input(UInt(8.W))
    val vpuOpcEnd = Input(UInt(8.W))

    val vpuIssueValid = Input(Bool())
    val vpuIssueQueue = Input(UInt(2.W))
    val vpuIssueSlot = Input(UInt(1.W))
    val vpuIssueKeep = Input(Bool())
    val vpuCompleteValid = Input(Bool())
    val vpuCompleteQueue = Input(UInt(2.W))
    val vpuCompleteSlot = Input(UInt(1.W))

    val gemLdReady = Output(Vec(entriesPerQueue, Bool()))
    val gemExReady = Output(Vec(entriesPerQueue, Bool()))
    val gemStReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuLdReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuExReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuStReady = Output(Vec(entriesPerQueue, Bool()))
  })

  val deps = Module(new FusionExtEntries(
    nSharers = 1,
    local_addr_t = localAddrType,
    reservation_station_entries_ld = entriesPerQueue,
    reservation_station_entries_ex = entriesPerQueue,
    reservation_station_entries_st = entriesPerQueue,
    res_max_per_type = entriesPerQueue,
    vpuEntriesLd = entriesPerQueue,
    vpuEntriesEx = entriesPerQueue,
    vpuEntriesSt = entriesPerQueue))

  private def operand(
      valid: Bool,
      start: UInt,
      end: UInt): UDValid[OpT] = {
    val result = Wire(UDValid(new OpT(localAddrType)))
    result.valid := valid
    result.bits.start := 0.U.asTypeOf(localAddrType)
    result.bits.start.is_acc_addr := true.B
    result.bits.start.data := start
    result.bits.end := result.bits.start
    result.bits.end.data := end
    result.bits.wraps_around := false.B
    result
  }

  private def driveClient(
      client: EntriesForDeps,
      allocValid: Bool,
      allocQueue: UInt,
      allocSlot: UInt,
      opaStart: UInt,
      opaEnd: UInt,
      opaIsDst: Bool,
      opbValid: Bool,
      opbStart: UInt,
      opbEnd: UInt,
      opcValid: Bool,
      opcStart: UInt,
      opcEnd: UInt,
      issueValid: Bool,
      issueQueue: UInt,
      issueSlot: UInt,
      issueKeep: Bool,
      completeValid: Bool,
      completeQueue: UInt,
      completeSlot: UInt): Unit = {
    client.alloc_entry.valid := allocValid
    client.alloc_entry.bits.alloc_id := Cat(allocQueue, allocSlot)
    client.alloc_entry.bits.opa := operand(true.B, opaStart, opaEnd)
    client.alloc_entry.bits.opb := operand(opbValid, opbStart, opbEnd)
    client.alloc_entry.bits.opc := operand(opcValid, opcStart, opcEnd)
    client.alloc_entry.bits.opa_is_dst := opaIsDst
    client.alloc_entry.bits.not_config := true.B

    Seq((0, client.issue_ld), (1, client.issue_ex),
      (2, client.issue_st)).foreach { case (queue, issue) =>
      issue.valid := issueValid && issueQueue === queue.U
      issue.bits.issue_id := issueSlot
      issue.bits.valid := issueKeep
    }
    client.complete_ld := Mux(
      completeValid && completeQueue === 0.U,
      UIntToOH(completeSlot, entriesPerQueue), 0.U)
    client.complete_ex := Mux(
      completeValid && completeQueue === 1.U,
      UIntToOH(completeSlot, entriesPerQueue), 0.U)
    client.complete_st := Mux(
      completeValid && completeQueue === 2.U,
      UIntToOH(completeSlot, entriesPerQueue), 0.U)
  }

  val gemmini = deps.io.in.head
  driveClient(gemmini,
    io.gemAllocValid, io.gemAllocQueue, io.gemAllocSlot,
    io.gemOpaStart, io.gemOpaEnd, io.gemOpaIsDst,
    io.gemOpbValid, io.gemOpbStart, io.gemOpbEnd,
    io.gemOpcValid, io.gemOpcStart, io.gemOpcEnd,
    io.gemIssueValid, io.gemIssueQueue, io.gemIssueSlot,
    io.gemIssueKeep,
    io.gemCompleteValid, io.gemCompleteQueue, io.gemCompleteSlot)

  val vpu = deps.io.vpu.get
  driveClient(vpu,
    io.vpuAllocValid, io.vpuAllocQueue, io.vpuAllocSlot,
    io.vpuOpaStart, io.vpuOpaEnd, io.vpuOpaIsDst,
    io.vpuOpbValid, io.vpuOpbStart, io.vpuOpbEnd,
    io.vpuOpcValid, io.vpuOpcStart, io.vpuOpcEnd,
    io.vpuIssueValid, io.vpuIssueQueue, io.vpuIssueSlot,
    io.vpuIssueKeep,
    io.vpuCompleteValid, io.vpuCompleteQueue, io.vpuCompleteSlot)

  io.gemLdReady := gemmini.ld_deps_ready
  io.gemExReady := gemmini.ex_deps_ready
  io.gemStReady := gemmini.st_deps_ready
  io.vpuLdReady := vpu.ld_deps_ready
  io.vpuExReady := vpu.ex_deps_ready
  io.vpuStReady := vpu.st_deps_ready
}

class FusionExtEntriesTester(c: FusionExtEntriesHarness)
    extends PeekPokeTester(c) {
  private val ldq = 0
  private val exq = 1
  private val stq = 2

  private def idle(): Unit = {
    poke(c.io.gemAllocValid, false)
    poke(c.io.gemAllocQueue, 0)
    poke(c.io.gemAllocSlot, 0)
    poke(c.io.gemOpaStart, 0)
    poke(c.io.gemOpaEnd, 0)
    poke(c.io.gemOpaIsDst, false)
    poke(c.io.gemOpbValid, false)
    poke(c.io.gemOpbStart, 0)
    poke(c.io.gemOpbEnd, 0)
    poke(c.io.gemOpcValid, false)
    poke(c.io.gemOpcStart, 0)
    poke(c.io.gemOpcEnd, 0)
    poke(c.io.gemIssueValid, false)
    poke(c.io.gemIssueQueue, 0)
    poke(c.io.gemIssueSlot, 0)
    poke(c.io.gemIssueKeep, true)
    poke(c.io.gemCompleteValid, false)
    poke(c.io.gemCompleteQueue, 0)
    poke(c.io.gemCompleteSlot, 0)

    poke(c.io.vpuAllocValid, false)
    poke(c.io.vpuAllocQueue, 0)
    poke(c.io.vpuAllocSlot, 0)
    poke(c.io.vpuOpaStart, 0)
    poke(c.io.vpuOpaEnd, 0)
    poke(c.io.vpuOpaIsDst, false)
    poke(c.io.vpuOpbValid, false)
    poke(c.io.vpuOpbStart, 0)
    poke(c.io.vpuOpbEnd, 0)
    poke(c.io.vpuOpcValid, false)
    poke(c.io.vpuOpcStart, 0)
    poke(c.io.vpuOpcEnd, 0)
    poke(c.io.vpuIssueValid, false)
    poke(c.io.vpuIssueQueue, 0)
    poke(c.io.vpuIssueSlot, 0)
    poke(c.io.vpuIssueKeep, true)
    poke(c.io.vpuCompleteValid, false)
    poke(c.io.vpuCompleteQueue, 0)
    poke(c.io.vpuCompleteSlot, 0)
  }

  private def allocateGem(
      queue: Int,
      slot: Int,
      start: Int,
      end: Int,
      isDst: Boolean,
      opb: Option[(Int, Int)] = None,
      opc: Option[(Int, Int)] = None): Unit = {
    poke(c.io.gemAllocValid, true)
    poke(c.io.gemAllocQueue, queue)
    poke(c.io.gemAllocSlot, slot)
    poke(c.io.gemOpaStart, start)
    poke(c.io.gemOpaEnd, end)
    poke(c.io.gemOpaIsDst, isDst)
    poke(c.io.gemOpbValid, opb.nonEmpty)
    opb.foreach { case (s, e) =>
      poke(c.io.gemOpbStart, s)
      poke(c.io.gemOpbEnd, e)
    }
    poke(c.io.gemOpcValid, opc.nonEmpty)
    opc.foreach { case (s, e) =>
      poke(c.io.gemOpcStart, s)
      poke(c.io.gemOpcEnd, e)
    }
    step(1)
    idle()
  }

  private def allocateVpu(
      queue: Int,
      slot: Int,
      start: Int,
      end: Int,
      isDst: Boolean,
      opb: Option[(Int, Int)] = None,
      opc: Option[(Int, Int)] = None): Unit = {
    poke(c.io.vpuAllocValid, true)
    poke(c.io.vpuAllocQueue, queue)
    poke(c.io.vpuAllocSlot, slot)
    poke(c.io.vpuOpaStart, start)
    poke(c.io.vpuOpaEnd, end)
    poke(c.io.vpuOpaIsDst, isDst)
    poke(c.io.vpuOpbValid, opb.nonEmpty)
    opb.foreach { case (s, e) =>
      poke(c.io.vpuOpbStart, s)
      poke(c.io.vpuOpbEnd, e)
    }
    poke(c.io.vpuOpcValid, opc.nonEmpty)
    opc.foreach { case (s, e) =>
      poke(c.io.vpuOpcStart, s)
      poke(c.io.vpuOpcEnd, e)
    }
    step(1)
    idle()
  }

  private def completeGem(queue: Int, slot: Int): Unit = {
    poke(c.io.gemCompleteValid, true)
    poke(c.io.gemCompleteQueue, queue)
    poke(c.io.gemCompleteSlot, slot)
    step(1)
    idle()
  }

  private def completeVpu(queue: Int, slot: Int): Unit = {
    poke(c.io.vpuCompleteValid, true)
    poke(c.io.vpuCompleteQueue, queue)
    poke(c.io.vpuCompleteSlot, slot)
    step(1)
    idle()
  }

  private def issueGemCompleteOnIssue(queue: Int, slot: Int): Unit = {
    poke(c.io.gemIssueValid, true)
    poke(c.io.gemIssueQueue, queue)
    poke(c.io.gemIssueSlot, slot)
    poke(c.io.gemIssueKeep, false)
    step(1)
    idle()
  }

  private def issueVpuCompleteOnIssue(queue: Int, slot: Int): Unit = {
    poke(c.io.vpuIssueValid, true)
    poke(c.io.vpuIssueQueue, queue)
    poke(c.io.vpuIssueSlot, slot)
    poke(c.io.vpuIssueKeep, false)
    step(1)
    idle()
  }

  idle()
  step(1)

  // Gemmini-to-Gemmini dependencies stay local and must not appear here.
  allocateGem(ldq, 0, 0, 4, isDst = true)
  allocateGem(exq, 0, 0, 4, isDst = false)
  expect(c.io.gemExReady(0), true)
  completeGem(exq, 0)
  completeGem(ldq, 0)

  // Gemmini writer -> VPU reader (RAW).
  allocateGem(ldq, 0, 8, 12, isDst = true)
  allocateVpu(exq, 0, 8, 12, isDst = false)
  expect(c.io.vpuExReady(0), false)
  completeGem(ldq, 0)
  expect(c.io.vpuExReady(0), true)
  completeVpu(exq, 0)

  // VPU writer -> Gemmini reader (RAW), the reverse owner direction.
  allocateVpu(ldq, 0, 16, 20, isDst = true)
  allocateGem(exq, 0, 16, 20, isDst = false)
  expect(c.io.gemExReady(0), false)
  completeVpu(ldq, 0)
  expect(c.io.gemExReady(0), true)
  completeGem(exq, 0)

  // A younger Gemmini LOAD writer waits for an older VPU EXECUTE reader (WAR).
  allocateVpu(exq, 0, 24, 28, isDst = false)
  allocateGem(ldq, 0, 24, 28, isDst = true)
  expect(c.io.gemLdReady(0), false)
  completeVpu(exq, 0)
  expect(c.io.gemLdReady(0), true)
  completeGem(ldq, 0)

  // Cross-owner EX/EX WAW is external even though both use the EX class.
  allocateGem(exq, 0, 32, 36, isDst = true)
  allocateVpu(exq, 0, 32, 36, isDst = true)
  expect(c.io.vpuExReady(0), false)
  completeGem(exq, 0)
  expect(c.io.vpuExReady(0), true)
  completeVpu(exq, 0)

  // Reverse EX/EX direction: a Gemmini reader waits for an older VPU writer.
  allocateVpu(exq, 0, 40, 44, isDst = true)
  allocateGem(exq, 0, 40, 44, isDst = false)
  expect(c.io.gemExReady(0), false)
  completeVpu(exq, 0)
  expect(c.io.gemExReady(0), true)
  completeGem(exq, 0)

  // VPU same-owner, different-class dependencies remain external.
  allocateVpu(ldq, 0, 48, 52, isDst = true)
  allocateVpu(exq, 0, 48, 52, isDst = false)
  expect(c.io.vpuExReady(0), false)
  completeVpu(ldq, 0)
  expect(c.io.vpuExReady(0), true)
  completeVpu(exq, 0)

  // The VPU's third operand (opc/source1) participates in RAW detection.
  allocateGem(ldq, 0, 64, 68, isDst = true)
  allocateVpu(exq, 0, 72, 76, isDst = true,
    opc = Some(64 -> 68))
  expect(c.io.vpuExReady(0), false)
  completeGem(ldq, 0)
  expect(c.io.vpuExReady(0), true)
  completeVpu(exq, 0)

  // A Gemmini complete-on-issue event releases an external VPU waiter.
  allocateGem(ldq, 0, 80, 84, isDst = true)
  allocateVpu(exq, 0, 80, 84, isDst = false)
  expect(c.io.vpuExReady(0), false)
  issueGemCompleteOnIssue(ldq, 0)
  expect(c.io.vpuExReady(0), true)
  completeVpu(exq, 0)

  // The symmetric VPU complete-on-issue path releases a Gemmini waiter.
  allocateVpu(ldq, 0, 88, 92, isDst = true)
  allocateGem(exq, 0, 88, 92, isDst = false)
  expect(c.io.gemExReady(0), false)
  issueVpuCompleteOnIssue(ldq, 0)
  expect(c.io.gemExReady(0), true)
  completeGem(exq, 0)

  // Pipelined VPU LOADs may complete out of order, so an overlapping younger
  // writer remains blocked until the older writer completes.
  allocateVpu(ldq, 0, 96, 100, isDst = true)
  allocateVpu(ldq, 1, 96, 100, isDst = true)
  expect(c.io.vpuLdReady(1), false)
  completeVpu(ldq, 0)
  expect(c.io.vpuLdReady(1), true)
  completeVpu(ldq, 1)

  // VPU EXECUTE is also pipelined across independent units. Preserve its
  // same-class RAW, WAR, and WAW edges through completion.
  allocateVpu(exq, 0, 104, 108, isDst = true)
  allocateVpu(exq, 1, 104, 108, isDst = false)
  expect(c.io.vpuExReady(1), false)
  completeVpu(exq, 0)
  expect(c.io.vpuExReady(1), true)
  completeVpu(exq, 1)

  allocateVpu(exq, 0, 112, 116, isDst = false)
  allocateVpu(exq, 1, 112, 116, isDst = true)
  expect(c.io.vpuExReady(1), false)
  completeVpu(exq, 0)
  expect(c.io.vpuExReady(1), true)
  completeVpu(exq, 1)

  allocateVpu(exq, 0, 120, 124, isDst = true)
  allocateVpu(exq, 1, 120, 124, isDst = true)
  expect(c.io.vpuExReady(1), false)
  completeVpu(exq, 0)
  expect(c.io.vpuExReady(1), true)
  completeVpu(exq, 1)

  // Two STORE entries only read the ACC, so same-class overlap is harmless.
  allocateVpu(stq, 0, 128, 132, isDst = false)
  allocateVpu(stq, 1, 128, 132, isDst = false)
  expect(c.io.vpuStReady(1), true)
  completeVpu(stq, 0)
  completeVpu(stq, 1)
}

class FusionExtEntriesUnitTest extends ChiselFlatSpec {
  behavior of "compact singleton Gemmini/VPU dependency tracking"

  it should "retain external and pipelined VPU address hazards" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/fusion-ext-entries")
    chisel3.iotesters.Driver.execute(args,
      () => new FusionExtEntriesHarness) { c =>
      new FusionExtEntriesTester(c)
    } should be(true)
  }
}
