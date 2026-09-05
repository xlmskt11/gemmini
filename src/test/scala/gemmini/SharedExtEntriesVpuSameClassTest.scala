package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.util.Cat

/** Focused lifecycle harness for VPU entries appended to SharedExtEntries.
  *
  * Gemmini is intentionally idle: these checks distinguish VPU same-owner,
  * same-queue dependency state from the already-covered cross-owner paths.
  */
class SharedExtEntriesVpuSameClassHarness extends Module {
  private val entries = 2
  private val localAddrType = new LocalAddr(
    sp_banks = 4, sp_bank_entries = 64,
    acc_banks = 4, acc_bank_entries = 64)

  val io = IO(new Bundle {
    val allocValid = Input(Bool())
    val allocQueue = Input(UInt(2.W))
    val allocSlot = Input(UInt(1.W))
    val allocStart = Input(UInt(8.W))
    val allocEnd = Input(UInt(8.W))
    val allocIsDst = Input(Bool())

    val issueValid = Input(Bool())
    val issueQueue = Input(UInt(2.W))
    val issueSlot = Input(UInt(1.W))
    val issueKeep = Input(Bool())

    val completeLd = Input(UInt(entries.W))
    val completeEx = Input(UInt(entries.W))
    val completeSt = Input(UInt(entries.W))

    val ldReady = Output(Vec(entries, Bool()))
    val exReady = Output(Vec(entries, Bool()))
    val stReady = Output(Vec(entries, Bool()))
  })

  val deps = Module(new SharedExtEntries(
    nSharers = 1,
    local_addr_t = localAddrType,
    reservation_station_entries_ld = entries,
    reservation_station_entries_ex = entries,
    reservation_station_entries_st = entries,
    res_max_per_type = entries,
    vpuEntriesLd = entries,
    vpuEntriesEx = entries,
    vpuEntriesSt = entries))

  val gemmini = deps.io.in.head
  gemmini.alloc_entry.valid := false.B
  gemmini.alloc_entry.bits := 0.U.asTypeOf(gemmini.alloc_entry.bits)
  gemmini.issue_ld := 0.U.asTypeOf(gemmini.issue_ld)
  gemmini.issue_ex := 0.U.asTypeOf(gemmini.issue_ex)
  gemmini.issue_st := 0.U.asTypeOf(gemmini.issue_st)
  gemmini.complete_ld := 0.U
  gemmini.complete_ex := 0.U
  gemmini.complete_st := 0.U

  val vpu = deps.io.vpu.get
  vpu.alloc_entry.valid := io.allocValid
  vpu.alloc_entry.bits.alloc_id := Cat(io.allocQueue, io.allocSlot)
  vpu.alloc_entry.bits.opa.valid := true.B
  vpu.alloc_entry.bits.opa.bits.start :=
    0.U.asTypeOf(vpu.alloc_entry.bits.opa.bits.start)
  vpu.alloc_entry.bits.opa.bits.start.is_acc_addr := true.B
  vpu.alloc_entry.bits.opa.bits.start.data := io.allocStart
  vpu.alloc_entry.bits.opa.bits.end :=
    vpu.alloc_entry.bits.opa.bits.start
  vpu.alloc_entry.bits.opa.bits.end.data := io.allocEnd
  vpu.alloc_entry.bits.opa.bits.wraps_around := false.B
  vpu.alloc_entry.bits.opb := 0.U.asTypeOf(vpu.alloc_entry.bits.opb)
  vpu.alloc_entry.bits.opc := 0.U.asTypeOf(vpu.alloc_entry.bits.opc)
  vpu.alloc_entry.bits.opa_is_dst := io.allocIsDst
  vpu.alloc_entry.bits.not_config := true.B

  vpu.issue_ld.valid := io.issueValid && io.issueQueue === 0.U
  vpu.issue_ld.bits.issue_id := io.issueSlot
  vpu.issue_ld.bits.valid := io.issueKeep
  vpu.issue_ex.valid := io.issueValid && io.issueQueue === 1.U
  vpu.issue_ex.bits.issue_id := io.issueSlot
  vpu.issue_ex.bits.valid := io.issueKeep
  vpu.issue_st.valid := io.issueValid && io.issueQueue === 2.U
  vpu.issue_st.bits.issue_id := io.issueSlot
  vpu.issue_st.bits.valid := io.issueKeep
  vpu.complete_ld := io.completeLd
  vpu.complete_ex := io.completeEx
  vpu.complete_st := io.completeSt

  io.ldReady := vpu.ld_deps_ready
  io.exReady := vpu.ex_deps_ready
  io.stReady := vpu.st_deps_ready
}

class SharedExtEntriesVpuSameClassTester(
    c: SharedExtEntriesVpuSameClassHarness) extends PeekPokeTester(c) {
  private val ldq = 0
  private val exq = 1
  private val stq = 2

  private def idle(): Unit = {
    poke(c.io.allocValid, false)
    poke(c.io.allocQueue, 0)
    poke(c.io.allocSlot, 0)
    poke(c.io.allocStart, 0)
    poke(c.io.allocEnd, 0)
    poke(c.io.allocIsDst, false)
    poke(c.io.issueValid, false)
    poke(c.io.issueQueue, 0)
    poke(c.io.issueSlot, 0)
    poke(c.io.issueKeep, true)
    poke(c.io.completeLd, 0)
    poke(c.io.completeEx, 0)
    poke(c.io.completeSt, 0)
  }

  private def allocate(queue: Int, slot: Int, start: Int, end: Int,
                       isDst: Boolean): Unit = {
    poke(c.io.allocValid, true)
    poke(c.io.allocQueue, queue)
    poke(c.io.allocSlot, slot)
    poke(c.io.allocStart, start)
    poke(c.io.allocEnd, end)
    poke(c.io.allocIsDst, isDst)
    step(1)
    idle()
  }

  private def issue(queue: Int, slot: Int): Unit = {
    poke(c.io.issueValid, true)
    poke(c.io.issueQueue, queue)
    poke(c.io.issueSlot, slot)
    poke(c.io.issueKeep, true)
    step(1)
    idle()
  }

  private def complete(queue: Int, mask: Int): Unit = {
    queue match {
      case `ldq` => poke(c.io.completeLd, mask)
      case `exq` => poke(c.io.completeEx, mask)
      case `stq` => poke(c.io.completeSt, mask)
    }
    step(1)
    idle()
  }

  idle()
  step(1)

  // Two loads write the same ACC interval. The younger WAW must survive the
  // producer's issue and clear only when its final SRAM write completes.
  allocate(ldq, slot = 0, start = 0, end = 4, isDst = true)
  allocate(ldq, slot = 1, start = 0, end = 4, isDst = true)
  expect(c.io.ldReady(0), true)
  expect(c.io.ldReady(1), false)
  issue(ldq, slot = 0)
  expect(c.io.ldReady(1), false)
  complete(ldq, mask = 1)
  expect(c.io.ldReady(1), true)
  issue(ldq, slot = 1)
  complete(ldq, mask = 2)

  // Disjoint loads are independent. Both may become issued and remain live
  // concurrently; the shared table must not add a FIFO-order dependency.
  allocate(ldq, slot = 0, start = 8, end = 12, isDst = true)
  allocate(ldq, slot = 1, start = 12, end = 16, isDst = true)
  expect(c.io.ldReady(0), true)
  expect(c.io.ldReady(1), true)
  issue(ldq, slot = 0)
  expect(c.io.ldReady(1), true)
  issue(ldq, slot = 1)
  complete(ldq, mask = 3)

  // Same-owner VPU EX/EX WAW is also completion-based. VPU commands can enter
  // independent units, so accepting the older command is not its commit.
  allocate(exq, slot = 0, start = 20, end = 24, isDst = true)
  allocate(exq, slot = 1, start = 20, end = 24, isDst = true)
  expect(c.io.exReady(0), true)
  expect(c.io.exReady(1), false)
  issue(exq, slot = 0)
  expect(c.io.exReady(1), false)
  complete(exq, mask = 1)
  expect(c.io.exReady(1), true)
  issue(exq, slot = 1)
  complete(exq, mask = 2)

  // Independent same-class execute commands retain overlap. This protects
  // the intended scalar/vector and distinct-unit pipelining.
  allocate(exq, slot = 0, start = 32, end = 36, isDst = true)
  allocate(exq, slot = 1, start = 40, end = 44, isDst = false)
  expect(c.io.exReady(0), true)
  expect(c.io.exReady(1), true)
  issue(exq, slot = 0)
  expect(c.io.exReady(1), true)
  issue(exq, slot = 1)
  complete(exq, mask = 3)

  // Store/store is read/read. Even an identical interval is intentionally
  // unblocked and both entries may be live at once.
  allocate(stq, slot = 0, start = 48, end = 52, isDst = false)
  allocate(stq, slot = 1, start = 48, end = 52, isDst = false)
  expect(c.io.stReady(0), true)
  expect(c.io.stReady(1), true)
  issue(stq, slot = 0)
  expect(c.io.stReady(1), true)
  issue(stq, slot = 1)
  complete(stq, mask = 3)
}

class SharedExtEntriesVpuSameClassUnitTest extends ChiselFlatSpec {
  behavior of "SharedExtEntries VPU same-class dependencies"

  it should "retain only real LD/EX hazards through producer completion" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/shared-ext-vpu-same-class")
    chisel3.iotesters.Driver.execute(args,
      () => new SharedExtEntriesVpuSameClassHarness) { c =>
      new SharedExtEntriesVpuSameClassTester(c)
    } should be(true)
  }
}
