package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.{Cat, UIntToOH}

class SharedExtEntriesAccExHarness extends Module {
  private val nSharers = 2
  private val entriesPerQueue = 2
  private val localAddrType = new LocalAddr(
    sp_banks = 4, sp_bank_entries = 64,
    acc_banks = 4, acc_bank_entries = 64)

  val io = IO(new Bundle {
    val alloc_valid = Input(Vec(nSharers, Bool()))
    val alloc_queue = Input(Vec(nSharers, UInt(2.W)))
    val alloc_slot = Input(Vec(nSharers, UInt(1.W)))
    val alloc_addr = Input(Vec(nSharers, UInt(8.W)))
    val alloc_opa_is_acc = Input(Vec(nSharers, Bool()))
    val alloc_is_dst = Input(Vec(nSharers, Bool()))
    val alloc_opb_valid = Input(Vec(nSharers, Bool()))
    val alloc_opb_addr = Input(Vec(nSharers, UInt(8.W)))

    val issue_valid = Input(Vec(nSharers, Bool()))
    val issue_queue = Input(Vec(nSharers, UInt(2.W)))
    val issue_keep = Input(Vec(nSharers, Bool()))
    val issue_slot = Input(Vec(nSharers, UInt(1.W)))
    val complete_valid = Input(Vec(nSharers, Bool()))
    val complete_queue = Input(Vec(nSharers, UInt(2.W)))
    val complete_slot = Input(Vec(nSharers, UInt(1.W)))

    val ld_ready = Output(Vec(nSharers,
      Vec(entriesPerQueue, Bool())))
    val ex_ready = Output(Vec(nSharers,
      Vec(entriesPerQueue, Bool())))
  })

  val deps = Module(new SharedExtEntries(
    nSharers = nSharers,
    local_addr_t = localAddrType,
    reservation_station_entries_ld = entriesPerQueue,
    reservation_station_entries_ex = entriesPerQueue,
    reservation_station_entries_st = entriesPerQueue,
    res_max_per_type = entriesPerQueue))

  for (i <- 0 until nSharers) {
    val start = WireInit(0.U.asTypeOf(localAddrType))
    start.is_acc_addr := io.alloc_opa_is_acc(i)
    start.data := io.alloc_addr(i)
    val opbStart = WireInit(0.U.asTypeOf(localAddrType))
    opbStart.is_acc_addr := true.B
    opbStart.data := io.alloc_opb_addr(i)

    deps.io.in(i).alloc_entry.valid := io.alloc_valid(i)
    deps.io.in(i).alloc_entry.bits.alloc_id :=
      Cat(io.alloc_queue(i), io.alloc_slot(i))
    deps.io.in(i).alloc_entry.bits.opa.valid := true.B
    deps.io.in(i).alloc_entry.bits.opa.bits.start := start
    deps.io.in(i).alloc_entry.bits.opa.bits.end := start + 1.U
    deps.io.in(i).alloc_entry.bits.opa.bits.wraps_around := false.B
    deps.io.in(i).alloc_entry.bits.opb.valid := io.alloc_opb_valid(i)
    deps.io.in(i).alloc_entry.bits.opb.bits.start := opbStart
    deps.io.in(i).alloc_entry.bits.opb.bits.end := opbStart + 1.U
    deps.io.in(i).alloc_entry.bits.opb.bits.wraps_around := false.B
    deps.io.in(i).alloc_entry.bits.opc :=
      0.U.asTypeOf(deps.io.in(i).alloc_entry.bits.opc)
    deps.io.in(i).alloc_entry.bits.opc.valid := false.B
    deps.io.in(i).alloc_entry.bits.opa_is_dst := io.alloc_is_dst(i)
    deps.io.in(i).alloc_entry.bits.not_config := true.B

    Seq((0, deps.io.in(i).issue_ld), (1, deps.io.in(i).issue_ex),
      (2, deps.io.in(i).issue_st)).foreach { case (queue, issue) =>
      issue.valid := io.issue_valid(i) && io.issue_queue(i) === queue.U
      issue.bits.issue_id := io.issue_slot(i)
      issue.bits.valid := io.issue_keep(i)
    }

    deps.io.in(i).complete_ld := Mux(
      io.complete_valid(i) && io.complete_queue(i) === 0.U,
      UIntToOH(io.complete_slot(i), entriesPerQueue), 0.U)
    deps.io.in(i).complete_ex := Mux(
      io.complete_valid(i) && io.complete_queue(i) === 1.U,
      UIntToOH(io.complete_slot(i), entriesPerQueue), 0.U)
    deps.io.in(i).complete_st := Mux(
      io.complete_valid(i) && io.complete_queue(i) === 2.U,
      UIntToOH(io.complete_slot(i), entriesPerQueue), 0.U)

    io.ld_ready(i) := deps.io.in(i).ld_deps_ready
    io.ex_ready(i) := deps.io.in(i).ex_deps_ready
  }
}

trait SharedExtEntriesAccExPokes {
  this: PeekPokeTester[SharedExtEntriesAccExHarness] =>

  protected def idle(c: SharedExtEntriesAccExHarness): Unit = {
    for (i <- 0 until 2) {
      poke(c.io.alloc_valid(i), false)
      poke(c.io.alloc_queue(i), 1)
      poke(c.io.alloc_slot(i), 0)
      poke(c.io.alloc_addr(i), 0)
      poke(c.io.alloc_opa_is_acc(i), true)
      poke(c.io.alloc_is_dst(i), true)
      poke(c.io.alloc_opb_valid(i), false)
      poke(c.io.alloc_opb_addr(i), 0)
      poke(c.io.issue_valid(i), false)
      poke(c.io.issue_queue(i), 1)
      poke(c.io.issue_keep(i), true)
      poke(c.io.issue_slot(i), 0)
      poke(c.io.complete_valid(i), false)
      poke(c.io.complete_queue(i), 1)
      poke(c.io.complete_slot(i), 0)
    }
  }
}

class SharedExtEntriesCrossExDependencyTester(
    c: SharedExtEntriesAccExHarness)
    extends PeekPokeTester(c) with SharedExtEntriesAccExPokes {

  private def allocate(sharer: Int, queue: Int, address: Int,
                       isDst: Boolean = true, slot: Int = 0): Unit = {
    idle(c)
    poke(c.io.alloc_valid(sharer), true)
    poke(c.io.alloc_queue(sharer), queue)
    poke(c.io.alloc_slot(sharer), slot)
    poke(c.io.alloc_addr(sharer), address)
    poke(c.io.alloc_is_dst(sharer), isDst)
    step(1)
    idle(c)
  }

  private def complete(sharer: Int, queue: Int, slot: Int = 0): Unit = {
    idle(c)
    poke(c.io.complete_valid(sharer), true)
    poke(c.io.complete_queue(sharer), queue)
    poke(c.io.complete_slot(sharer), slot)
    step(1)
    idle(c)
  }

  private def completeBoth(queue: Int): Unit = {
    idle(c)
    for (sharer <- 0 until 2) {
      poke(c.io.complete_valid(sharer), true)
      poke(c.io.complete_queue(sharer), queue)
    }
    step(1)
    idle(c)
  }

  idle(c)

  // Same-Gemmini EX ordering belongs to its local ReservationStation. The
  // external cross-EX table must not recreate a dependency on that owner's
  // own EX slice.
  allocate(sharer = 0, queue = 1, address = 6, slot = 0)
  allocate(sharer = 0, queue = 1, address = 6, slot = 1)
  expect(c.io.ex_ready(0)(1), true)
  complete(sharer = 0, queue = 1, slot = 0)
  complete(sharer = 0, queue = 1, slot = 1)

  // Local EX queue order does not cover another Gemmini's reservation station.
  // An overlapping cross-Gemmini EX writer therefore waits for completion,
  // rather than merely for issue, of the older EX writer.
  allocate(sharer = 0, queue = 1, address = 7)
  allocate(sharer = 1, queue = 1, address = 7)
  expect(c.io.ex_ready(1)(0), false)

  idle(c)
  poke(c.io.issue_valid(0), true)
  poke(c.io.issue_queue(0), 1)
  step(1)
  idle(c)
  expect(c.io.ex_ready(1)(0), false)

  complete(sharer = 0, queue = 1)
  expect(c.io.ex_ready(1)(0), true)
  complete(sharer = 1, queue = 1)

  allocate(sharer = 0, queue = 0, address = 8)
  allocate(sharer = 1, queue = 0, address = 8)
  expect(c.io.ld_ready(1)(0), true)
  completeBoth(queue = 0)

  // Same-cycle non-EX allocations retain their previous behavior. EX conflicts
  // have a separate assertion because the Valid-only ports cannot assign age.
  for (sharer <- 0 until 2) {
    poke(c.io.alloc_valid(sharer), true)
    poke(c.io.alloc_queue(sharer), 0)
    poke(c.io.alloc_addr(sharer), 9)
  }
  step(1)
  idle(c)
  expect(c.io.ld_ready(0)(0), true)
  expect(c.io.ld_ready(1)(0), true)
  completeBoth(queue = 0)

  // Existing cross-queue RAW/WAW ordering remains: an EX touching an older
  // LD destination cannot issue until that LD completes.
  allocate(sharer = 0, queue = 0, address = 10)
  allocate(sharer = 1, queue = 1, address = 10, isDst = false)
  expect(c.io.ex_ready(1)(0), false)

  idle(c)
  poke(c.io.issue_valid(0), true)
  poke(c.io.issue_queue(0), 0)
  step(1)
  idle(c)
  expect(c.io.ex_ready(1)(0), false)

  complete(sharer = 0, queue = 0)
  expect(c.io.ex_ready(1)(0), true)
  complete(sharer = 1, queue = 1)

  // The reverse EX-to-LD dependency is retained as well.
  allocate(sharer = 0, queue = 1, address = 11)
  allocate(sharer = 1, queue = 0, address = 11)
  expect(c.io.ld_ready(1)(0), false)
  complete(sharer = 0, queue = 1)
  expect(c.io.ld_ready(1)(0), true)
}

class SharedExtEntriesSameCycleAccExConflictTester(
    c: SharedExtEntriesAccExHarness)
    extends PeekPokeTester(c) with SharedExtEntriesAccExPokes {
  idle(c)
  for (sharer <- 0 until 2) {
    poke(c.io.alloc_valid(sharer), true)
    poke(c.io.alloc_queue(sharer), 1)
    poke(c.io.alloc_addr(sharer), 12)
    poke(c.io.alloc_is_dst(sharer), true)
  }
  step(1)
}

class SharedExtEntriesAccExDependencyUnitTest extends ChiselFlatSpec {
  behavior of "SharedExtEntries cross-Gemmini EX ordering"

  it should "retain overlapping cross-Gemmini EX edges until completion" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-cross-ex-ordering")

    chisel3.iotesters.Driver.execute(args,
      () => new SharedExtEntriesAccExHarness) {
      c => new SharedExtEntriesCrossExDependencyTester(c)
    } should be(true)
  }

  it should "reject conflicting cross-Gemmini EX allocations in one cycle" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-same-cycle-ex-conflict")

    val stopped = intercept[treadle.executable.StopException] {
      chisel3.iotesters.Driver.execute(args,
        () => new SharedExtEntriesAccExHarness) {
        c => new SharedExtEntriesSameCycleAccExConflictTester(c)
      }
    }
    stopped.getMessage should include("deps.assert")
  }
}
