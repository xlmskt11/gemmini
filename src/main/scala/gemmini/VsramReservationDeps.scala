package gemmini

import chisel3._
import chisel3.util._
import Util._

/** One half-open VSRAM row interval. */
class VsramAccess(addressBits: Int) extends Bundle {
  val valid = Bool()
  val start = UInt(addressBits.W)
  val end = UInt(addressBits.W)
  val wraps_around = Bool()
  val read = Bool()
  val write = Bool()

  private def nonEmpty: Bool = wraps_around || start =/= end

  private def contains(point: UInt): Bool = Mux(wraps_around,
    point >= start || point < end,
    point >= start && point < end)

  def overlaps(other: VsramAccess): Bool =
    valid && other.valid && nonEmpty && other.nonEmpty &&
      (contains(other.start) || other.contains(start))

  def conflicts(other: VsramAccess): Bool =
    overlaps(other) &&
      ((write && (other.read || other.write)) ||
        (read && other.write))
}

object VsramAccess {
  val maxVpuAccesses = 3

  /** Convert a Gemmini LocalAddr interval to the matching VSRAM row interval. */
  def fromLocal(op: UDValid[OpT], enabled: Bool,
                vsramRows: Int, read: Bool, write: Bool): VsramAccess = {
    val addressBits = op.bits.start.data.getWidth
    val result = Wire(new VsramAccess(addressBits))
    val exactBoundary = op.bits.end.data === vsramRows.U
    val wraps = !exactBoundary &&
      (op.bits.end.data > vsramRows.U ||
        op.bits.end.data < op.bits.start.data)

    result.valid := op.valid && enabled && !op.bits.start.is_garbage()
    result.start := op.bits.start.full_acc_addr().pad(addressBits)
    result.end := Mux(exactBoundary, vsramRows.U(addressBits.W),
      op.bits.end.full_acc_addr().pad(addressBits))
    result.wraps_around := wraps
    result.read := read
    result.write := write
    result
  }
}

class VsramAllocEntry(addressBits: Int, resMaxPerType: Int)
    extends Bundle {
  val alloc_id = UInt((log2Up(resMaxPerType) + 2).W)
  val access = new VsramAccess(addressBits)
}

/** Gemmini-facing side-table port. It intentionally mirrors EntriesForDeps. */
class VsramEntriesForDeps(
  localAddr: LocalAddr,
  reservationStationEntriesLd: Int,
  reservationStationEntriesEx: Int,
  reservationStationEntriesSt: Int,
  resMaxPerType: Int) extends Bundle {

  val alloc_entry = Output(Valid(new VsramAllocEntry(
    localAddr.data.getWidth, resMaxPerType)))

  val ld_deps_ready = Input(Vec(reservationStationEntriesLd, Bool()))
  val ex_deps_ready = Input(Vec(reservationStationEntriesEx, Bool()))
  val st_deps_ready = Input(Vec(reservationStationEntriesSt, Bool()))

  val issue_ld = Output(Valid(new IssueEvent(resMaxPerType)))
  val issue_ex = Output(Valid(new IssueEvent(resMaxPerType)))
  val issue_st = Output(Valid(new IssueEvent(resMaxPerType)))
  val complete_id = Output(Valid(UInt((log2Up(resMaxPerType) + 2).W)))
}

class VsramSlotAllocation(
  addressBits: Int,
  localEntries: Int,
  maxAccesses: Int = VsramAccess.maxVpuAccesses) extends Bundle {
  val slot = UInt((1 max log2Ceil(localEntries)).W)
  val accesses = Vec(maxAccesses, new VsramAccess(addressBits))
}

/** VPU-facing port. VPU completion is a mask because its engines retire in parallel. */
class VsramClientIO(
  addressBits: Int,
  localEntries: Int,
  maxAccesses: Int = VsramAccess.maxVpuAccesses) extends Bundle {
  val allocate = Output(Valid(new VsramSlotAllocation(
    addressBits, localEntries, maxAccesses)))
  val ready = Input(Vec(localEntries, Bool()))
  val release = Output(UInt(localEntries.W))
}

/**
  * VSRAM dependency side table between Gemmini and the VPU.
  *
  * Gemmini-to-Gemmini SPAD/ACC ordering remains in SharedExtEntries. VPU-to-VPU
  * ordering remains in VpuReservationStation. This module stores only the
  * cross-accelerator VSRAM edges, so a Gemmini entry has VPU dependencies and
  * a VPU entry has Gemmini LD/EX dependencies. Gemmini STORE has no VSRAM
  * fusion operation today, but its ready/issue ports are retained to match the
  * existing ReservationDeps interface.
  *
  * The four Gemmini matrix ports use disjoint software-owned VSRAM ranges.
  * Consequently Gemmini-to-Gemmini VSRAM conflicts are outside this table.
  */
class VsramExtEntries(
  nSharers: Int,
  localAddr: LocalAddr,
  reservationStationEntriesLd: Int,
  reservationStationEntriesEx: Int,
  reservationStationEntriesSt: Int,
  resMaxPerType: Int,
  vpuEntries: Int,
  maxVpuAccesses: Int = VsramAccess.maxVpuAccesses) extends Module {

  require(nSharers > 0)
  require(vpuEntries > 0)

  private val addressBits = localAddr.data.getWidth
  private val totalLdEntries = reservationStationEntriesLd * nSharers
  private val totalExEntries = reservationStationEntriesEx * nSharers
  private val typeWidth = log2Up(resMaxPerType)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new VsramEntriesForDeps(
      localAddr, reservationStationEntriesLd, reservationStationEntriesEx,
      reservationStationEntriesSt, resMaxPerType)))
    val vpu = Flipped(new VsramClientIO(
      addressBits, vpuEntries, maxVpuAccesses))
  })

  val ldq :: exq :: stq :: Nil = Enum(3)

  class GemminiEntry extends Bundle {
    val access = new VsramAccess(addressBits)
    val deps_vpu = Vec(vpuEntries, Bool())

    def ready(dummy: Int = 0): Bool = !deps_vpu.reduce(_ || _)
  }

  class VpuEntry extends Bundle {
    val accesses = Vec(maxVpuAccesses, new VsramAccess(addressBits))
    val deps_ld = Vec(totalLdEntries, Bool())
    val deps_ex = Vec(totalExEntries, Bool())

    def ready(dummy: Int = 0): Bool =
      !(deps_ld.reduce(_ || _) || deps_ex.reduce(_ || _))
  }

  val total_entries_ld = Reg(Vec(totalLdEntries,
    UDValid(new GemminiEntry)))
  val total_entries_ex = Reg(Vec(totalExEntries,
    UDValid(new GemminiEntry)))
  val total_entries_vpu = Reg(Vec(vpuEntries,
    UDValid(new VpuEntry)))

  private def conflicts(access: VsramAccess,
                        accesses: Vec[VsramAccess]): Bool =
    accesses.map(access.conflicts).reduce(_ || _)

  val release_ld = WireInit(VecInit(Seq.fill(totalLdEntries)(false.B)))
  val release_ex = WireInit(VecInit(Seq.fill(totalExEntries)(false.B)))

  // Generate Gemmini release masks from the same typed IDs used by
  // SharedExtEntries. complete-on-issue is included for interface symmetry;
  // current VSRAM commands are all completion-retired.
  for (sharer <- 0 until nSharers) {
    val completed = io.in(sharer).complete_id
    val completedQueue = completed.bits(typeWidth + 1, typeWidth)
    val completedId = completed.bits(typeWidth - 1, 0)

    when(completed.valid && completedQueue === ldq) {
      release_ld((sharer * reservationStationEntriesLd).U +& completedId) := true.B
    }
    when(completed.valid && completedQueue === exq) {
      release_ex((sharer * reservationStationEntriesEx).U +& completedId) := true.B
    }

    when(io.in(sharer).issue_ld.valid &&
        !io.in(sharer).issue_ld.bits.valid) {
      release_ld((sharer * reservationStationEntriesLd).U +&
        io.in(sharer).issue_ld.bits.issue_id) := true.B
    }
    when(io.in(sharer).issue_ex.valid &&
        !io.in(sharer).issue_ex.bits.valid) {
      release_ex((sharer * reservationStationEntriesEx).U +&
        io.in(sharer).issue_ex.bits.issue_id) := true.B
    }
  }

  val live_ld = VecInit(total_entries_ld.zip(release_ld).map {
    case (entry, released) => entry.valid && !released
  })
  val live_ex = VecInit(total_entries_ex.zip(release_ex).map {
    case (entry, released) => entry.valid && !released
  })
  val live_vpu = VecInit(total_entries_vpu.zipWithIndex.map {
    case (entry, slot) => entry.valid && !io.vpu.release(slot)
  })

  // Completion removes both the producer and every dependency on it.
  total_entries_ld.zip(release_ld).foreach { case (entry, released) =>
    entry.bits.deps_vpu.zipWithIndex.foreach { case (dep, slot) =>
      when(io.vpu.release(slot)) { dep := false.B }
    }
    when(released) { entry.valid := false.B }
  }
  total_entries_ex.zip(release_ex).foreach { case (entry, released) =>
    entry.bits.deps_vpu.zipWithIndex.foreach { case (dep, slot) =>
      when(io.vpu.release(slot)) { dep := false.B }
    }
    when(released) { entry.valid := false.B }
  }
  total_entries_vpu.zipWithIndex.foreach { case (entry, slot) =>
    entry.bits.deps_ld.zip(release_ld).foreach { case (dep, released) =>
      when(released) { dep := false.B }
    }
    entry.bits.deps_ex.zip(release_ex).foreach { case (dep, released) =>
      when(released) { dep := false.B }
    }
    when(io.vpu.release(slot)) { entry.valid := false.B }
  }

  // Allocate Gemmini VSRAM accesses. Each Gemmini owns a fixed slice, exactly
  // as in SharedExtEntries, so no flattened LD/EX/ST slot is exposed to its RS.
  for (sharer <- 0 until nSharers) {
    val allocation = io.in(sharer).alloc_entry
    val queue = allocation.bits.alloc_id(typeWidth + 1, typeWidth)
    val localId = allocation.bits.alloc_id(typeWidth - 1, 0)

    when(allocation.valid && queue === ldq) {
      val globalId = (sharer * reservationStationEntriesLd).U +& localId
      assert(localId < reservationStationEntriesLd.U)
      assert(!live_ld(globalId),
        "VSRAM dependency allocation overwrote a live Gemmini LD slot")
      total_entries_ld(globalId).valid := true.B
      total_entries_ld(globalId).bits.access := allocation.bits.access
      total_entries_ld(globalId).bits.deps_vpu := VecInit(
        total_entries_vpu.zip(live_vpu).map { case (entry, live) =>
          live && conflicts(allocation.bits.access, entry.bits.accesses)
        })
    }

    when(allocation.valid && queue === exq) {
      val globalId = (sharer * reservationStationEntriesEx).U +& localId
      assert(localId < reservationStationEntriesEx.U)
      assert(!live_ex(globalId),
        "VSRAM dependency allocation overwrote a live Gemmini EX slot")
      total_entries_ex(globalId).valid := true.B
      total_entries_ex(globalId).bits.access := allocation.bits.access
      total_entries_ex(globalId).bits.deps_vpu := VecInit(
        total_entries_vpu.zip(live_vpu).map { case (entry, live) =>
          live && conflicts(allocation.bits.access, entry.bits.accesses)
        })
    }
  }

  // A same-cycle VPU allocation is younger than Gemmini 0..n allocations.
  // Include those new Gemmini accesses as dependencies even though their
  // registers become visible only after this edge.
  def sameCycleGemminiConflict(
      globalId: Int, queue: UInt, entriesPerSharer: Int): Bool = {
    val sharer = globalId / entriesPerSharer
    val localId = globalId % entriesPerSharer
    val allocation = io.in(sharer).alloc_entry
    val allocationQueue = allocation.bits.alloc_id(
      typeWidth + 1, typeWidth)
    val allocationId = allocation.bits.alloc_id(typeWidth - 1, 0)
    allocation.valid && allocationQueue === queue &&
      allocationId === localId.U &&
      conflicts(allocation.bits.access, io.vpu.allocate.bits.accesses)
  }

  when(io.vpu.allocate.valid) {
    val slot = io.vpu.allocate.bits.slot
    val hasAccess = io.vpu.allocate.bits.accesses.map(_.valid).reduce(_ || _)
    assert(slot < vpuEntries.U)
    assert(!live_vpu(slot),
      "VSRAM dependency allocation overwrote a live VPU slot")

    for (localSlot <- 0 until vpuEntries) {
      when(slot === localSlot.U) {
        total_entries_vpu(localSlot).valid := hasAccess
        total_entries_vpu(localSlot).bits.accesses :=
          io.vpu.allocate.bits.accesses
        total_entries_vpu(localSlot).bits.deps_ld := VecInit(
          total_entries_ld.zip(live_ld).zipWithIndex.map {
            case ((entry, live), globalId) =>
              (live && conflicts(entry.bits.access,
                io.vpu.allocate.bits.accesses)) ||
                sameCycleGemminiConflict(
                  globalId, ldq, reservationStationEntriesLd)
          })
        total_entries_vpu(localSlot).bits.deps_ex := VecInit(
          total_entries_ex.zip(live_ex).zipWithIndex.map {
            case ((entry, live), globalId) =>
              (live && conflicts(entry.bits.access,
                io.vpu.allocate.bits.accesses)) ||
                sameCycleGemminiConflict(
                  globalId, exq, reservationStationEntriesEx)
          })
      }
    }
  }

  for (sharer <- 0 until nSharers) {
    val ldSlice = total_entries_ld.slice(
      sharer * reservationStationEntriesLd,
      (sharer + 1) * reservationStationEntriesLd)
    val exSlice = total_entries_ex.slice(
      sharer * reservationStationEntriesEx,
      (sharer + 1) * reservationStationEntriesEx)

    io.in(sharer).ld_deps_ready := ldSlice.map(entry =>
      !entry.valid || entry.bits.ready())
    io.in(sharer).ex_deps_ready := exSlice.map(entry =>
      !entry.valid || entry.bits.ready())
    io.in(sharer).st_deps_ready := VecInit(
      Seq.fill(reservationStationEntriesSt)(true.B))
  }

  io.vpu.ready := total_entries_vpu.map(entry =>
    !entry.valid || entry.bits.ready())

  when(reset.asBool) {
    total_entries_ld.foreach(_.valid := false.B)
    total_entries_ex.foreach(_.valid := false.B)
    total_entries_vpu.foreach(_.valid := false.B)
  }
}
