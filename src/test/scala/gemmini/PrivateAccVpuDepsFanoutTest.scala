package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.Cat

class PrivateAccVpuDepsFanoutHarness extends Module {
  private val nOwners = 4
  private val rowsPerOwner = 64
  private val entriesPerQueue = 2
  private val globalAddressBits = 9
  private val localAddrType = new LocalAddr(
    sp_banks = 4, sp_bank_entries = 64,
    acc_banks = 4, acc_bank_entries = 16)

  val io = IO(new Bundle {
    val allocValid = Input(Bool())
    val allocQueue = Input(UInt(2.W))
    val allocSlot = Input(UInt(1.W))
    val opaValid = Input(Bool())
    val opaStart = Input(UInt(globalAddressBits.W))
    val opaEnd = Input(UInt(globalAddressBits.W))
    val opaWrap = Input(Bool())
    val opaIsDst = Input(Bool())
    val opbValid = Input(Bool())
    val opbStart = Input(UInt(globalAddressBits.W))
    val opbEnd = Input(UInt(globalAddressBits.W))
    val opbWrap = Input(Bool())
    val opcValid = Input(Bool())
    val opcStart = Input(UInt(globalAddressBits.W))
    val opcEnd = Input(UInt(globalAddressBits.W))
    val opcWrap = Input(Bool())

    val issueLdValid = Input(Bool())
    val issueLdId = Input(UInt(1.W))
    val issueLdKeep = Input(Bool())
    val issueExValid = Input(Bool())
    val issueExId = Input(UInt(1.W))
    val issueExKeep = Input(Bool())
    val issueStValid = Input(Bool())
    val issueStId = Input(UInt(1.W))
    val issueStKeep = Input(Bool())
    val completeLd = Input(UInt(entriesPerQueue.W))
    val completeEx = Input(UInt(entriesPerQueue.W))
    val completeSt = Input(UInt(entriesPerQueue.W))

    val ownerLdReady = Input(Vec(nOwners,
      Vec(entriesPerQueue, Bool())))
    val ownerExReady = Input(Vec(nOwners,
      Vec(entriesPerQueue, Bool())))
    val ownerStReady = Input(Vec(nOwners,
      Vec(entriesPerQueue, Bool())))

    val vpuLdReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuExReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuStReady = Output(Vec(entriesPerQueue, Bool()))

    val ownerAllocValid = Output(Vec(nOwners, Bool()))
    val ownerAllocId = Output(Vec(nOwners, UInt(3.W)))
    val ownerOpaValid = Output(Vec(nOwners, Bool()))
    val ownerOpaStart = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpaEnd = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpaWrap = Output(Vec(nOwners, Bool()))
    val ownerOpbValid = Output(Vec(nOwners, Bool()))
    val ownerOpbStart = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpbEnd = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpbWrap = Output(Vec(nOwners, Bool()))
    val ownerOpcValid = Output(Vec(nOwners, Bool()))
    val ownerOpcStart = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpcEnd = Output(Vec(nOwners, UInt(6.W)))
    val ownerOpcWrap = Output(Vec(nOwners, Bool()))
    val ownerIssueLdValid = Output(Vec(nOwners, Bool()))
    val ownerIssueExValid = Output(Vec(nOwners, Bool()))
    val ownerIssueStValid = Output(Vec(nOwners, Bool()))
    val ownerIssueExId = Output(Vec(nOwners, UInt(1.W)))
    val ownerIssueExKeep = Output(Vec(nOwners, Bool()))
    val ownerCompleteLd = Output(Vec(nOwners,
      UInt(entriesPerQueue.W)))
    val ownerCompleteEx = Output(Vec(nOwners,
      UInt(entriesPerQueue.W)))
    val ownerCompleteSt = Output(Vec(nOwners,
      UInt(entriesPerQueue.W)))
  })

  val fanout = Module(new PrivateAccVpuDepsFanout(
    localAddrType = localAddrType,
    nOwners = nOwners,
    rowsPerOwner = rowsPerOwner,
    globalAddressBits = globalAddressBits,
    reservationStationEntriesLd = entriesPerQueue,
    reservationStationEntriesEx = entriesPerQueue,
    reservationStationEntriesSt = entriesPerQueue,
    resMaxPerType = entriesPerQueue))

  val vpu = fanout.io.vpu
  vpu.alloc_entry.valid := io.allocValid
  vpu.alloc_entry.bits.alloc_id := Cat(io.allocQueue, io.allocSlot)
  vpu.alloc_entry.bits.opa.valid := io.opaValid
  vpu.alloc_entry.bits.opa.start := io.opaStart
  vpu.alloc_entry.bits.opa.end := io.opaEnd
  vpu.alloc_entry.bits.opa.wraps_around := io.opaWrap
  vpu.alloc_entry.bits.opb.valid := io.opbValid
  vpu.alloc_entry.bits.opb.start := io.opbStart
  vpu.alloc_entry.bits.opb.end := io.opbEnd
  vpu.alloc_entry.bits.opb.wraps_around := io.opbWrap
  vpu.alloc_entry.bits.opc.valid := io.opcValid
  vpu.alloc_entry.bits.opc.start := io.opcStart
  vpu.alloc_entry.bits.opc.end := io.opcEnd
  vpu.alloc_entry.bits.opc.wraps_around := io.opcWrap
  vpu.alloc_entry.bits.opa_is_dst := io.opaIsDst

  vpu.issue_ld.valid := io.issueLdValid
  vpu.issue_ld.bits.issue_id := io.issueLdId
  vpu.issue_ld.bits.valid := io.issueLdKeep
  vpu.issue_ex.valid := io.issueExValid
  vpu.issue_ex.bits.issue_id := io.issueExId
  vpu.issue_ex.bits.valid := io.issueExKeep
  vpu.issue_st.valid := io.issueStValid
  vpu.issue_st.bits.issue_id := io.issueStId
  vpu.issue_st.bits.valid := io.issueStKeep
  vpu.complete_ld := io.completeLd
  vpu.complete_ex := io.completeEx
  vpu.complete_st := io.completeSt

  io.vpuLdReady := vpu.ld_deps_ready
  io.vpuExReady := vpu.ex_deps_ready
  io.vpuStReady := vpu.st_deps_ready

  for ((owner, ownerId) <- fanout.io.owners.zipWithIndex) {
    owner.ld_deps_ready := io.ownerLdReady(ownerId)
    owner.ex_deps_ready := io.ownerExReady(ownerId)
    owner.st_deps_ready := io.ownerStReady(ownerId)

    io.ownerAllocValid(ownerId) := owner.alloc_entry.valid
    io.ownerAllocId(ownerId) := owner.alloc_entry.bits.alloc_id
    io.ownerOpaValid(ownerId) := owner.alloc_entry.bits.opa.valid
    io.ownerOpaStart(ownerId) :=
      owner.alloc_entry.bits.opa.bits.start.full_acc_addr()
    io.ownerOpaEnd(ownerId) :=
      owner.alloc_entry.bits.opa.bits.end.full_acc_addr()
    io.ownerOpaWrap(ownerId) :=
      owner.alloc_entry.bits.opa.bits.wraps_around
    io.ownerOpbValid(ownerId) := owner.alloc_entry.bits.opb.valid
    io.ownerOpbStart(ownerId) :=
      owner.alloc_entry.bits.opb.bits.start.full_acc_addr()
    io.ownerOpbEnd(ownerId) :=
      owner.alloc_entry.bits.opb.bits.end.full_acc_addr()
    io.ownerOpbWrap(ownerId) :=
      owner.alloc_entry.bits.opb.bits.wraps_around
    io.ownerOpcValid(ownerId) := owner.alloc_entry.bits.opc.valid
    io.ownerOpcStart(ownerId) :=
      owner.alloc_entry.bits.opc.bits.start.full_acc_addr()
    io.ownerOpcEnd(ownerId) :=
      owner.alloc_entry.bits.opc.bits.end.full_acc_addr()
    io.ownerOpcWrap(ownerId) :=
      owner.alloc_entry.bits.opc.bits.wraps_around

    io.ownerIssueLdValid(ownerId) := owner.issue_ld.valid
    io.ownerIssueExValid(ownerId) := owner.issue_ex.valid
    io.ownerIssueStValid(ownerId) := owner.issue_st.valid
    io.ownerIssueExId(ownerId) := owner.issue_ex.bits.issue_id
    io.ownerIssueExKeep(ownerId) := owner.issue_ex.bits.valid
    io.ownerCompleteLd(ownerId) := owner.complete_ld
    io.ownerCompleteEx(ownerId) := owner.complete_ex
    io.ownerCompleteSt(ownerId) := owner.complete_st
  }
}

trait PrivateAccVpuDepsFanoutPokes {
  this: PeekPokeTester[PrivateAccVpuDepsFanoutHarness] =>

  protected def idle(c: PrivateAccVpuDepsFanoutHarness): Unit = {
    poke(c.io.allocValid, false)
    poke(c.io.allocQueue, 0)
    poke(c.io.allocSlot, 0)
    poke(c.io.opaValid, false)
    poke(c.io.opaStart, 0)
    poke(c.io.opaEnd, 0)
    poke(c.io.opaWrap, false)
    poke(c.io.opaIsDst, false)
    poke(c.io.opbValid, false)
    poke(c.io.opbStart, 0)
    poke(c.io.opbEnd, 0)
    poke(c.io.opbWrap, false)
    poke(c.io.opcValid, false)
    poke(c.io.opcStart, 0)
    poke(c.io.opcEnd, 0)
    poke(c.io.opcWrap, false)
    poke(c.io.issueLdValid, false)
    poke(c.io.issueLdId, 0)
    poke(c.io.issueLdKeep, false)
    poke(c.io.issueExValid, false)
    poke(c.io.issueExId, 0)
    poke(c.io.issueExKeep, false)
    poke(c.io.issueStValid, false)
    poke(c.io.issueStId, 0)
    poke(c.io.issueStKeep, false)
    poke(c.io.completeLd, 0)
    poke(c.io.completeEx, 0)
    poke(c.io.completeSt, 0)
    for (owner <- 0 until 4; entry <- 0 until 2) {
      poke(c.io.ownerLdReady(owner)(entry), true)
      poke(c.io.ownerExReady(owner)(entry), true)
      poke(c.io.ownerStReady(owner)(entry), true)
    }
  }
}

class PrivateAccVpuDepsFanoutTester(c: PrivateAccVpuDepsFanoutHarness)
    extends PeekPokeTester(c) with PrivateAccVpuDepsFanoutPokes {
  idle(c)

  poke(c.io.allocValid, true)
  poke(c.io.allocQueue, 1)
  poke(c.io.allocSlot, 1)
  poke(c.io.opaValid, true)
  poke(c.io.opaStart, 70)
  poke(c.io.opaEnd, 75)
  poke(c.io.opaIsDst, true)
  poke(c.io.opbValid, true)
  poke(c.io.opbStart, 128)
  poke(c.io.opbEnd, 192)
  poke(c.io.opcValid, true)
  poke(c.io.opcStart, 255)
  poke(c.io.opcEnd, 256)
  poke(c.io.opcWrap, true)

  for (owner <- 0 until 4) {
    expect(c.io.ownerAllocValid(owner), true)
    expect(c.io.ownerAllocId(owner), 3)
    expect(c.io.ownerOpaValid(owner), owner == 1)
    expect(c.io.ownerOpbValid(owner), owner == 2)
    expect(c.io.ownerOpcValid(owner), owner == 3)
  }
  expect(c.io.ownerOpaStart(1), 6)
  expect(c.io.ownerOpaEnd(1), 11)
  expect(c.io.ownerOpaWrap(1), false)
  expect(c.io.ownerOpbStart(2), 0)
  expect(c.io.ownerOpbEnd(2), 0)
  expect(c.io.ownerOpbWrap(2), true)
  expect(c.io.ownerOpcStart(3), 63)
  expect(c.io.ownerOpcEnd(3), 0)
  expect(c.io.ownerOpcWrap(3), true)

  // One architectural interval may span several private accumulators. Each
  // owner receives only its local intersection.
  poke(c.io.opaStart, 60)
  poke(c.io.opaEnd, 132)
  poke(c.io.opbValid, false)
  poke(c.io.opcValid, false)
  expect(c.io.ownerOpaValid(0), true)
  expect(c.io.ownerOpaStart(0), 60)
  expect(c.io.ownerOpaEnd(0), 0)
  expect(c.io.ownerOpaWrap(0), true)
  expect(c.io.ownerOpaValid(1), true)
  expect(c.io.ownerOpaStart(1), 0)
  expect(c.io.ownerOpaEnd(1), 0)
  expect(c.io.ownerOpaWrap(1), true)
  expect(c.io.ownerOpaValid(2), true)
  expect(c.io.ownerOpaStart(2), 0)
  expect(c.io.ownerOpaEnd(2), 4)
  expect(c.io.ownerOpaWrap(2), false)
  expect(c.io.ownerOpaValid(3), false)

  // One blocked owner blocks the corresponding global VPU slot only.
  poke(c.io.ownerExReady(2)(1), false)
  expect(c.io.vpuExReady(0), true)
  expect(c.io.vpuExReady(1), false)
  expect(c.io.vpuLdReady(1), true)
  expect(c.io.vpuStReady(1), true)

  // Lifecycle signals are mirrored even to owners without a valid operand.
  poke(c.io.issueExValid, true)
  poke(c.io.issueExId, 1)
  poke(c.io.issueExKeep, true)
  poke(c.io.completeLd, 1)
  poke(c.io.completeEx, 2)
  poke(c.io.completeSt, 3)
  for (owner <- 0 until 4) {
    expect(c.io.ownerIssueLdValid(owner), false)
    expect(c.io.ownerIssueExValid(owner), true)
    expect(c.io.ownerIssueStValid(owner), false)
    expect(c.io.ownerIssueExId(owner), 1)
    expect(c.io.ownerIssueExKeep(owner), true)
    expect(c.io.ownerCompleteLd(owner), 1)
    expect(c.io.ownerCompleteEx(owner), 2)
    expect(c.io.ownerCompleteSt(owner), 3)
  }
}

class PrivateAccVpuDepsOutOfRangeTester(
    c: PrivateAccVpuDepsFanoutHarness)
    extends PeekPokeTester(c) with PrivateAccVpuDepsFanoutPokes {
  idle(c)
  poke(c.io.allocValid, true)
  poke(c.io.opaValid, true)
  poke(c.io.opaStart, 250)
  poke(c.io.opaEnd, 260)
  step(1)
}

class PrivateAccVpuDepsFanoutUnitTest extends ChiselFlatSpec {
  behavior of "private-ACC VPU dependency fanout"

  it should "localize operands and preserve one mirrored lifecycle" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/private-acc-vpu-deps-fanout")
    chisel3.iotesters.Driver.execute(args,
      () => new PrivateAccVpuDepsFanoutHarness) { c =>
      new PrivateAccVpuDepsFanoutTester(c)
    } should be(true)
  }

  it should "reject an operand interval escaping all private owners" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/private-acc-vpu-deps-out-of-range")
    val stopped = intercept[treadle.executable.StopException] {
      chisel3.iotesters.Driver.execute(args,
        () => new PrivateAccVpuDepsFanoutHarness) { c =>
        new PrivateAccVpuDepsOutOfRangeTester(c)
      }
    }
    stopped.getMessage should include("assert")
  }
}
