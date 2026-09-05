package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.{Cat, Mux1H, UIntToOH}

class SharedExtVpuAccHarness extends Module {
  private val nSharers = 2
  private val entriesPerQueue = 2
  private val vpuEntriesLd = 2
  private val vpuEntriesEx = 2
  private val vpuEntriesSt = 2
  private val localAddrType = new LocalAddr(
    sp_banks = 4, sp_bank_entries = 64,
    acc_banks = 4, acc_bank_entries = 64)

  val io = IO(new Bundle {
    val gemAllocValid = Input(Bool())
    val gemSharer = Input(UInt(1.W))
    val gemQueue = Input(UInt(2.W))
    val gemSlot = Input(UInt(1.W))
    val gemOpaStart = Input(UInt(8.W))
    val gemOpaEnd = Input(UInt(8.W))
    val gemOpaIsAcc = Input(Bool())
    val gemOpaIsDst = Input(Bool())
    val gemOpbValid = Input(Bool())
    val gemOpbStart = Input(UInt(8.W))
    val gemOpbEnd = Input(UInt(8.W))

    val gemIssueValid = Input(Bool())
    val gemIssueKeep = Input(Bool())
    val gemCompleteValid = Input(Bool())
    val gemCompleteQueue = Input(UInt(2.W))
    val gemCompleteSlot = Input(UInt(1.W))

    val vpuAllocValid = Input(Bool())
    val vpuAllocQueue = Input(UInt(2.W))
    val vpuAllocSlot = Input(UInt(1.W))
    val vpuStart = Input(UInt(8.W))
    val vpuEnd = Input(UInt(8.W))
    val vpuRead = Input(Bool())
    val vpuWrite = Input(Bool())
    val vpuOpbValid = Input(Bool())
    val vpuOpbStart = Input(UInt(8.W))
    val vpuOpbEnd = Input(UInt(8.W))
    val vpuOpcValid = Input(Bool())
    val vpuOpcStart = Input(UInt(8.W))
    val vpuOpcEnd = Input(UInt(8.W))
    val vpuCompleteLd = Input(UInt(vpuEntriesLd.W))
    val vpuCompleteEx = Input(UInt(vpuEntriesEx.W))
    val vpuCompleteSt = Input(UInt(vpuEntriesSt.W))

    val gemLdReady = Output(Vec(entriesPerQueue, Bool()))
    val gemExReady = Output(Vec(entriesPerQueue, Bool()))
    val gemStReady = Output(Vec(entriesPerQueue, Bool()))
    val vpuLdReady = Output(Vec(vpuEntriesLd, Bool()))
    val vpuExReady = Output(Vec(vpuEntriesEx, Bool()))
    val vpuStReady = Output(Vec(vpuEntriesSt, Bool()))
  })

  val deps = Module(new SharedExtEntries(
    nSharers = nSharers,
    local_addr_t = localAddrType,
    reservation_station_entries_ld = entriesPerQueue,
    reservation_station_entries_ex = entriesPerQueue,
    reservation_station_entries_st = entriesPerQueue,
    res_max_per_type = entriesPerQueue,
    vpuEntriesLd = vpuEntriesLd,
    vpuEntriesEx = vpuEntriesEx,
    vpuEntriesSt = vpuEntriesSt))

  val opaStart = WireInit(0.U.asTypeOf(localAddrType))
  opaStart.is_acc_addr := io.gemOpaIsAcc
  opaStart.data := io.gemOpaStart
  val opaEnd = WireInit(opaStart)
  opaEnd.data := io.gemOpaEnd
  val opbStart = WireInit(0.U.asTypeOf(localAddrType))
  opbStart.is_acc_addr := true.B
  opbStart.data := io.gemOpbStart
  val opbEnd = WireInit(opbStart)
  opbEnd.data := io.gemOpbEnd

  for ((gemmini, sharer) <- deps.io.in.zipWithIndex) {
    val selected = io.gemSharer === sharer.U
    gemmini.alloc_entry.valid := io.gemAllocValid && selected
    gemmini.alloc_entry.bits.alloc_id := Cat(io.gemQueue, io.gemSlot)
    gemmini.alloc_entry.bits.opa.valid := true.B
    gemmini.alloc_entry.bits.opa.bits.start := opaStart
    gemmini.alloc_entry.bits.opa.bits.end := opaEnd
    gemmini.alloc_entry.bits.opa.bits.wraps_around := false.B
    gemmini.alloc_entry.bits.opb.valid := io.gemOpbValid
    gemmini.alloc_entry.bits.opb.bits.start := opbStart
    gemmini.alloc_entry.bits.opb.bits.end := opbEnd
    gemmini.alloc_entry.bits.opb.bits.wraps_around := false.B
    gemmini.alloc_entry.bits.opc :=
      0.U.asTypeOf(gemmini.alloc_entry.bits.opc)
    gemmini.alloc_entry.bits.opc.valid := false.B
    gemmini.alloc_entry.bits.opa_is_dst := io.gemOpaIsDst
    gemmini.alloc_entry.bits.not_config := true.B

    Seq((0, gemmini.issue_ld), (1, gemmini.issue_ex),
      (2, gemmini.issue_st)).foreach { case (queue, issue) =>
      issue.valid := io.gemIssueValid && selected && io.gemQueue === queue.U
      issue.bits.issue_id := io.gemSlot
      issue.bits.valid := io.gemIssueKeep
    }
    gemmini.complete_ld := Mux(
      io.gemCompleteValid && selected && io.gemCompleteQueue === 0.U,
      UIntToOH(io.gemCompleteSlot, entriesPerQueue), 0.U)
    gemmini.complete_ex := Mux(
      io.gemCompleteValid && selected && io.gemCompleteQueue === 1.U,
      UIntToOH(io.gemCompleteSlot, entriesPerQueue), 0.U)
    gemmini.complete_st := Mux(
      io.gemCompleteValid && selected && io.gemCompleteQueue === 2.U,
      UIntToOH(io.gemCompleteSlot, entriesPerQueue), 0.U)
  }

  val vpuAdapter = Module(new VpuEntriesForDepsAdapter(
    localAddrType,
    addressBits = 8,
    reservationStationEntriesLd = vpuEntriesLd,
    reservationStationEntriesEx = vpuEntriesEx,
    reservationStationEntriesSt = vpuEntriesSt,
    resMaxPerType = entriesPerQueue))
  vpuAdapter.io.shared <> deps.io.vpu.get
  val vpu = vpuAdapter.io.vpu
  vpu.alloc_entry.valid := io.vpuAllocValid
  vpu.alloc_entry.bits.alloc_id := Cat(io.vpuAllocQueue, io.vpuAllocSlot)
  vpu.alloc_entry.bits.opa.valid := io.vpuRead || io.vpuWrite
  vpu.alloc_entry.bits.opa.start := io.vpuStart
  vpu.alloc_entry.bits.opa.end := io.vpuEnd
  vpu.alloc_entry.bits.opa.wraps_around := false.B
  vpu.alloc_entry.bits.opb.valid := io.vpuOpbValid
  vpu.alloc_entry.bits.opb.start := io.vpuOpbStart
  vpu.alloc_entry.bits.opb.end := io.vpuOpbEnd
  vpu.alloc_entry.bits.opb.wraps_around := false.B
  vpu.alloc_entry.bits.opc.valid := io.vpuOpcValid
  vpu.alloc_entry.bits.opc.start := io.vpuOpcStart
  vpu.alloc_entry.bits.opc.end := io.vpuOpcEnd
  vpu.alloc_entry.bits.opc.wraps_around := false.B
  vpu.alloc_entry.bits.opa_is_dst := io.vpuWrite
  vpu.issue_ld := 0.U.asTypeOf(vpu.issue_ld)
  vpu.issue_ex := 0.U.asTypeOf(vpu.issue_ex)
  vpu.issue_st := 0.U.asTypeOf(vpu.issue_st)
  vpu.complete_ld := io.vpuCompleteLd
  vpu.complete_ex := io.vpuCompleteEx
  vpu.complete_st := io.vpuCompleteSt

  io.gemLdReady := Mux1H(UIntToOH(io.gemSharer, nSharers),
    deps.io.in.map(_.ld_deps_ready))
  io.gemExReady := Mux1H(UIntToOH(io.gemSharer, nSharers),
    deps.io.in.map(_.ex_deps_ready))
  io.gemStReady := Mux1H(UIntToOH(io.gemSharer, nSharers),
    deps.io.in.map(_.st_deps_ready))
  io.vpuLdReady := vpu.ld_deps_ready
  io.vpuExReady := vpu.ex_deps_ready
  io.vpuStReady := vpu.st_deps_ready
}

trait SharedExtVpuAccPokes {
  this: PeekPokeTester[SharedExtVpuAccHarness] =>

  protected val ldq = 0
  protected val exq = 1
  protected val stq = 2

  protected def idle(c: SharedExtVpuAccHarness): Unit = {
    poke(c.io.gemAllocValid, false)
    poke(c.io.gemSharer, 0)
    poke(c.io.gemQueue, 0)
    poke(c.io.gemSlot, 0)
    poke(c.io.gemOpaStart, 0)
    poke(c.io.gemOpaEnd, 0)
    poke(c.io.gemOpaIsAcc, true)
    poke(c.io.gemOpaIsDst, false)
    poke(c.io.gemOpbValid, false)
    poke(c.io.gemOpbStart, 0)
    poke(c.io.gemOpbEnd, 0)
    poke(c.io.gemIssueValid, false)
    poke(c.io.gemIssueKeep, true)
    poke(c.io.gemCompleteValid, false)
    poke(c.io.gemCompleteQueue, 0)
    poke(c.io.gemCompleteSlot, 0)
    poke(c.io.vpuAllocValid, false)
    poke(c.io.vpuAllocQueue, 0)
    poke(c.io.vpuAllocSlot, 0)
    poke(c.io.vpuStart, 0)
    poke(c.io.vpuEnd, 0)
    poke(c.io.vpuRead, false)
    poke(c.io.vpuWrite, false)
    poke(c.io.vpuOpbValid, false)
    poke(c.io.vpuOpbStart, 0)
    poke(c.io.vpuOpbEnd, 0)
    poke(c.io.vpuOpcValid, false)
    poke(c.io.vpuOpcStart, 0)
    poke(c.io.vpuOpcEnd, 0)
    poke(c.io.vpuCompleteLd, 0)
    poke(c.io.vpuCompleteEx, 0)
    poke(c.io.vpuCompleteSt, 0)
  }

  protected def allocateGemmini(c: SharedExtVpuAccHarness,
                                queue: Int, slot: Int,
                                start: Int, end: Int,
                                isDst: Boolean,
                                opaIsAcc: Boolean = true,
                                opb: Option[(Int, Int)] = None,
                                sharer: Int = 0): Unit = {
    poke(c.io.gemAllocValid, true)
    poke(c.io.gemSharer, sharer)
    poke(c.io.gemQueue, queue)
    poke(c.io.gemSlot, slot)
    poke(c.io.gemOpaStart, start)
    poke(c.io.gemOpaEnd, end)
    poke(c.io.gemOpaIsAcc, opaIsAcc)
    poke(c.io.gemOpaIsDst, isDst)
    poke(c.io.gemOpbValid, opb.nonEmpty)
    opb.foreach { case (opbStart, opbEnd) =>
      poke(c.io.gemOpbStart, opbStart)
      poke(c.io.gemOpbEnd, opbEnd)
    }
    step(1)
    idle(c)
  }

  protected def completeGemmini(c: SharedExtVpuAccHarness,
                                queue: Int, slot: Int,
                                sharer: Int = 0): Unit = {
    poke(c.io.gemCompleteValid, true)
    poke(c.io.gemSharer, sharer)
    poke(c.io.gemCompleteQueue, queue)
    poke(c.io.gemCompleteSlot, slot)
    step(1)
    idle(c)
  }

  protected def allocateVpu(c: SharedExtVpuAccHarness,
                            queue: Int, start: Int, end: Int,
                            read: Boolean, write: Boolean,
                            opb: Option[(Int, Int)] = None,
                            opc: Option[(Int, Int)] = None,
                            slot: Int = 0): Unit = {
    poke(c.io.vpuAllocValid, true)
    poke(c.io.vpuAllocQueue, queue)
    poke(c.io.vpuAllocSlot, slot)
    poke(c.io.vpuStart, start)
    poke(c.io.vpuEnd, end)
    poke(c.io.vpuRead, read)
    poke(c.io.vpuWrite, write)
    poke(c.io.vpuOpbValid, opb.nonEmpty)
    opb.foreach { case (sourceStart, sourceEnd) =>
      poke(c.io.vpuOpbStart, sourceStart)
      poke(c.io.vpuOpbEnd, sourceEnd)
    }
    poke(c.io.vpuOpcValid, opc.nonEmpty)
    opc.foreach { case (sourceStart, sourceEnd) =>
      poke(c.io.vpuOpcStart, sourceStart)
      poke(c.io.vpuOpcEnd, sourceEnd)
    }
    step(1)
    idle(c)
  }

  protected def completeVpu(c: SharedExtVpuAccHarness,
                            queue: Int, slot: Int = 0): Unit = {
    queue match {
      case `ldq` => poke(c.io.vpuCompleteLd, 1 << slot)
      case `exq` => poke(c.io.vpuCompleteEx, 1 << slot)
      case `stq` => poke(c.io.vpuCompleteSt, 1 << slot)
    }
    step(1)
    idle(c)
  }

  protected def expectVpuReady(c: SharedExtVpuAccHarness,
                               queue: Int, expected: Boolean,
                               slot: Int = 0): Unit = {
    queue match {
      case `ldq` => expect(c.io.vpuLdReady(slot), expected)
      case `exq` => expect(c.io.vpuExReady(slot), expected)
      case `stq` => expect(c.io.vpuStReady(slot), expected)
    }
  }
}

class SharedExtVpuAccDependencyTester(c: SharedExtVpuAccHarness)
    extends PeekPokeTester(c) with SharedExtVpuAccPokes {
  idle(c)
  step(1)

  // A younger VPU STORE reader waits for an already-live Gemmini EXECUTE
  // writer because the accesses belong to different queues.
  allocateGemmini(c, exq, 0, 0, 4, isDst = true)
  allocateVpu(c, stq, 0, 4, read = true, write = false)
  expectVpuReady(c, stq, expected = false)
  completeGemmini(c, exq, 0)
  expectVpuReady(c, stq, expected = true)
  completeVpu(c, stq)

  // Flattened Gemmini slices use the owning sharer's completion ID. The VPU
  // EXECUTE reader is in a different queue from the Gemmini LOAD writer.
  allocateGemmini(c, ldq, 0, 4, 8, isDst = true, sharer = 1)
  allocateVpu(c, exq, 4, 8, read = true, write = false)
  expectVpuReady(c, exq, expected = false)
  completeGemmini(c, ldq, 0, sharer = 1)
  expectVpuReady(c, exq, expected = true)
  completeVpu(c, exq)

  // Gemmini and VPU own different reservation stations, so an overlapping
  // cross-owner EXECUTE pair is ordered by the shared dependency table.
  allocateGemmini(c, exq, 0, 84, 88, isDst = true)
  allocateVpu(c, exq, 84, 88, read = true, write = false)
  expectVpuReady(c, exq, expected = false)
  completeGemmini(c, exq, 0)
  expectVpuReady(c, exq, expected = true)
  completeVpu(c, exq)

  // The reverse allocation order uses the symmetric dependency path.
  allocateVpu(c, exq, 88, 92, read = false, write = true)
  allocateGemmini(c, exq, 0, 88, 92, isDst = false)
  expect(c.io.gemExReady(0), false)
  completeVpu(c, exq)
  expect(c.io.gemExReady(0), true)
  completeGemmini(c, exq, 0)

  // A younger Gemmini STORE waits for an already-live VPU writer. STORE is
  // now an ordinary ACC reader rather than a VSRAM-special-case omission.
  allocateVpu(c, ldq, 8, 12, read = false, write = true)
  allocateGemmini(c, stq, 0, 8, 12, isDst = false)
  expect(c.io.gemStReady(0), false)
  completeVpu(c, ldq)
  expect(c.io.gemStReady(0), true)
  completeGemmini(c, stq, 0)

  // opb participates independently: a compute with SPAD opa and ACC opb must
  // wait for the older VPU writer.
  allocateVpu(c, ldq, 16, 20, read = false, write = true)
  allocateGemmini(c, exq, 0, 32, 36, isDst = false,
    opaIsAcc = false, opb = Some(16 -> 20))
  expect(c.io.gemExReady(0), false)
  completeVpu(c, ldq)
  expect(c.io.gemExReady(0), true)
  completeGemmini(c, exq, 0)

  // VPU binary source1 is carried by opc. It must create the same RAW edge
  // as source0 when an older Gemmini LOAD writes that ACC range.
  allocateGemmini(c, ldq, 0, 52, 56, isDst = true)
  allocateVpu(c, exq, 60, 64, read = false, write = true,
    opb = Some(64 -> 68), opc = Some(52 -> 56))
  expectVpuReady(c, exq, expected = false)
  completeGemmini(c, ldq, 0)
  expectVpuReady(c, exq, expected = true)
  completeVpu(c, exq)

  // Fusion VPU-to-VPU ordering is also external: the younger overlapping
  // reader remains blocked until the older writer commits/releases.
  allocateVpu(c, ldq, 24, 28, read = false, write = true)
  allocateVpu(c, exq, 24, 28, read = true, write = false)
  expectVpuReady(c, exq, expected = false)
  completeVpu(c, ldq)
  expectVpuReady(c, exq, expected = true)
  completeVpu(c, exq)

  // Multiple older VPU writers in LOAD and EXECUTE produce independent edges
  // for a younger Gemmini STORE, which is a third queue.
  allocateVpu(c, ldq, 40, 44, read = false, write = true)
  allocateVpu(c, exq, 40, 44, read = false, write = true)
  allocateGemmini(c, stq, 0, 40, 44, isDst = false)
  expect(c.io.gemStReady(0), false)
  completeVpu(c, ldq)
  expect(c.io.gemStReady(0), false)
  completeVpu(c, exq)
  expect(c.io.gemStReady(0), true)
  completeGemmini(c, stq, 0)

  // Match Gemmini: complete the old occupant first, then recycle its slot on
  // the following cycle. The replacement is younger than the live Gemmini
  // reader and therefore acquires the reverse dependency.
  allocateVpu(c, ldq, 80, 84, read = false, write = true)
  allocateGemmini(c, exq, 0, 80, 84, isDst = false)
  expect(c.io.gemExReady(0), false)
  completeVpu(c, ldq)
  expect(c.io.gemExReady(0), true)
  allocateVpu(c, ldq, 80, 84, read = false, write = true)
  expectVpuReady(c, ldq, expected = false)
  completeGemmini(c, exq, 0)
  expectVpuReady(c, ldq, expected = true)
  completeVpu(c, ldq)

  // Same-owner VPU queues remain pipelined after issue. Keep LOAD WAW and
  // EXECUTE RAW/WAR/WAW edges until the older entry completes.
  allocateVpu(c, ldq, 96, 100, read = false, write = true, slot = 0)
  allocateVpu(c, ldq, 96, 100, read = false, write = true, slot = 1)
  expectVpuReady(c, ldq, expected = false, slot = 1)
  completeVpu(c, ldq, slot = 0)
  expectVpuReady(c, ldq, expected = true, slot = 1)
  completeVpu(c, ldq, slot = 1)

  allocateVpu(c, exq, 104, 108, read = false, write = true, slot = 0)
  allocateVpu(c, exq, 104, 108, read = true, write = false, slot = 1)
  expectVpuReady(c, exq, expected = false, slot = 1)
  completeVpu(c, exq, slot = 0)
  expectVpuReady(c, exq, expected = true, slot = 1)
  completeVpu(c, exq, slot = 1)

  allocateVpu(c, exq, 112, 116, read = true, write = false, slot = 0)
  allocateVpu(c, exq, 112, 116, read = false, write = true, slot = 1)
  expectVpuReady(c, exq, expected = false, slot = 1)
  completeVpu(c, exq, slot = 0)
  expectVpuReady(c, exq, expected = true, slot = 1)
  completeVpu(c, exq, slot = 1)

  allocateVpu(c, exq, 120, 124, read = false, write = true, slot = 0)
  allocateVpu(c, exq, 120, 124, read = false, write = true, slot = 1)
  expectVpuReady(c, exq, expected = false, slot = 1)
  completeVpu(c, exq, slot = 0)
  expectVpuReady(c, exq, expected = true, slot = 1)
  completeVpu(c, exq, slot = 1)

  allocateVpu(c, stq, 128, 132, read = true, write = false, slot = 0)
  allocateVpu(c, stq, 128, 132, read = true, write = false, slot = 1)
  expectVpuReady(c, stq, expected = true, slot = 1)
  completeVpu(c, stq, slot = 0)
  completeVpu(c, stq, slot = 1)
}

class SharedExtVpuAccSameCycleSafeTester(c: SharedExtVpuAccHarness)
    extends PeekPokeTester(c) with SharedExtVpuAccPokes {
  idle(c)
  step(1)

  // Same-cycle read/read overlap is allowed and creates no arbitrary edge.
  poke(c.io.gemAllocValid, true)
  poke(c.io.gemQueue, exq)
  poke(c.io.gemSlot, 0)
  poke(c.io.gemOpaStart, 48)
  poke(c.io.gemOpaEnd, 52)
  poke(c.io.gemOpaIsAcc, true)
  poke(c.io.gemOpaIsDst, false)
  poke(c.io.vpuAllocValid, true)
  poke(c.io.vpuAllocQueue, exq)
  poke(c.io.vpuAllocSlot, 0)
  poke(c.io.vpuStart, 48)
  poke(c.io.vpuEnd, 52)
  poke(c.io.vpuRead, true)
  step(1)
  idle(c)
  expect(c.io.gemExReady(0), true)
  expectVpuReady(c, exq, expected = true)
  completeGemmini(c, exq, 0)
  completeVpu(c, exq)

  // A disjoint writer/read pair is likewise accepted without a dependency.
  poke(c.io.gemAllocValid, true)
  poke(c.io.gemQueue, exq)
  poke(c.io.gemOpaStart, 56)
  poke(c.io.gemOpaEnd, 60)
  poke(c.io.gemOpaIsDst, true)
  poke(c.io.vpuAllocValid, true)
  poke(c.io.vpuAllocQueue, exq)
  poke(c.io.vpuStart, 64)
  poke(c.io.vpuEnd, 68)
  poke(c.io.vpuRead, true)
  step(1)
  idle(c)
  expect(c.io.gemExReady(0), true)
  expectVpuReady(c, exq, expected = true)
}

class SharedExtVpuAccSameCycleConflictTester(c: SharedExtVpuAccHarness)
    extends PeekPokeTester(c) with SharedExtVpuAccPokes {
  idle(c)
  poke(c.io.gemAllocValid, true)
  poke(c.io.gemQueue, exq)
  poke(c.io.gemSlot, 0)
  poke(c.io.gemOpaStart, 72)
  poke(c.io.gemOpaEnd, 76)
  poke(c.io.gemOpaIsAcc, true)
  poke(c.io.gemOpaIsDst, true)
  poke(c.io.vpuAllocValid, true)
  poke(c.io.vpuAllocQueue, exq)
  poke(c.io.vpuAllocSlot, 0)
  poke(c.io.vpuStart, 72)
  poke(c.io.vpuEnd, 76)
  poke(c.io.vpuRead, true)
  step(1)
}

class AccReservationDepsUnitTest extends ChiselFlatSpec {
  behavior of "unified ACC dependency tracking"

  it should "order live Gemmini and VPU ACC accesses in both directions" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-vpu-acc")
    chisel3.iotesters.Driver.execute(args,
      () => new SharedExtVpuAccHarness) { c =>
      new SharedExtVpuAccDependencyTester(c)
    } should be(true)
  }

  it should "allow safe same-cycle Gemmini/VPU allocations without edges" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-vpu-acc-same-cycle-safe")
    chisel3.iotesters.Driver.execute(args,
      () => new SharedExtVpuAccHarness) { c =>
      new SharedExtVpuAccSameCycleSafeTester(c)
    } should be(true)
  }

  it should "reject a same-cycle overlapping cross-owner EX pair" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-vpu-acc-same-cycle-conflict")
    val stopped = intercept[treadle.executable.StopException] {
      chisel3.iotesters.Driver.execute(args,
        () => new SharedExtVpuAccHarness) { c =>
        new SharedExtVpuAccSameCycleConflictTester(c)
      }
    }
    stopped.getMessage should include("deps.assert")
  }
}
