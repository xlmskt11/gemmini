package gemmini

import chisel3._
import chisel3.util._
import Util._

/** Row interval exported by the VPU before it is cast to Gemmini's LocalAddr. */
class VpuDepOperand(addressBits: Int) extends Bundle {
    val valid = Bool()
    val start = UInt(addressBits.W)
    val end = UInt(addressBits.W)
    val wraps_around = Bool()
}

/** The VPU equivalent of AllocEntry, expressed in shared-ACC row addresses. */
class VpuDepAllocEntry(addressBits: Int, resMaxPerType: Int) extends Bundle {
    val alloc_id = UInt((log2Up(resMaxPerType) + 2).W)
    val opa = new VpuDepOperand(addressBits)
    val opb = new VpuDepOperand(addressBits)
    val opc = new VpuDepOperand(addressBits)
    val opa_is_dst = Bool()
}

/**
  * VPU-side lifecycle port. It mirrors EntriesForDeps. Only its operand
  * address representation differs; the adapter below performs the one
  * required shared-ACC-row to LocalAddr conversion.
  */
class VpuEntriesForDeps(
    addressBits: Int,
    reservationStationEntriesLd: Int,
    reservationStationEntriesEx: Int,
    reservationStationEntriesSt: Int,
    resMaxPerType: Int) extends Bundle {
    val alloc_entry = Output(Valid(new VpuDepAllocEntry(
        addressBits, resMaxPerType)))

    val ld_deps_ready = Input(Vec(reservationStationEntriesLd, Bool()))
    val ex_deps_ready = Input(Vec(reservationStationEntriesEx, Bool()))
    val st_deps_ready = Input(Vec(reservationStationEntriesSt, Bool()))

    val issue_ld = Output(Valid(new IssueEvent(resMaxPerType)))
    val issue_ex = Output(Valid(new IssueEvent(resMaxPerType)))
    val issue_st = Output(Valid(new IssueEvent(resMaxPerType)))

    val complete_ld = Output(UInt(reservationStationEntriesLd.W))
    val complete_ex = Output(UInt(reservationStationEntriesEx.W))
    val complete_st = Output(UInt(reservationStationEntriesSt.W))
}

/** Convert only the VPU address representation; dependency state stays shared. */
class VpuEntriesForDepsAdapter(
    localAddrType: LocalAddr,
    addressBits: Int,
    reservationStationEntriesLd: Int,
    reservationStationEntriesEx: Int,
    reservationStationEntriesSt: Int,
    resMaxPerType: Int) extends Module {
    private val vpuResMaxPerType = math.max(reservationStationEntriesLd,
        math.max(reservationStationEntriesEx, reservationStationEntriesSt))
    require(log2Up(vpuResMaxPerType) == log2Up(resMaxPerType),
        "VPU and shared dependency IDs must use the same local-ID width")

    val io = IO(new Bundle {
        val vpu = Flipped(new VpuEntriesForDeps(
            addressBits,
            reservationStationEntriesLd,
            reservationStationEntriesEx,
            reservationStationEntriesSt,
            resMaxPerType))
        val shared = new EntriesForDeps(
            localAddrType,
            reservationStationEntriesLd,
            reservationStationEntriesEx,
            reservationStationEntriesSt,
            resMaxPerType)
    })

    private def accAddress(row: UInt): LocalAddr = {
        val result = WireInit(0.U.asTypeOf(localAddrType))
        result.is_acc_addr := true.B
        result.data := row
        result
    }

    private def convert(in: VpuDepOperand): UDValid[OpT] = {
        val result = Wire(UDValid(new OpT(localAddrType)))
        result.valid := in.valid
        result.bits.start := accAddress(in.start)
        result.bits.end := accAddress(in.end)
        result.bits.wraps_around := in.wraps_around
        result
    }

    io.shared.alloc_entry.valid := io.vpu.alloc_entry.valid
    io.shared.alloc_entry.bits.alloc_id := io.vpu.alloc_entry.bits.alloc_id
    io.shared.alloc_entry.bits.opa := convert(io.vpu.alloc_entry.bits.opa)
    io.shared.alloc_entry.bits.opb := convert(io.vpu.alloc_entry.bits.opb)
    io.shared.alloc_entry.bits.opc := convert(io.vpu.alloc_entry.bits.opc)
    io.shared.alloc_entry.bits.opa_is_dst :=
        io.vpu.alloc_entry.bits.opa_is_dst
    io.shared.alloc_entry.bits.not_config := true.B

    io.vpu.ld_deps_ready := io.shared.ld_deps_ready
    io.vpu.ex_deps_ready := io.shared.ex_deps_ready
    io.vpu.st_deps_ready := io.shared.st_deps_ready

    io.shared.issue_ld := io.vpu.issue_ld
    io.shared.issue_ex := io.vpu.issue_ex
    io.shared.issue_st := io.vpu.issue_st
    io.shared.complete_ld := io.vpu.complete_ld
    io.shared.complete_ex := io.vpu.complete_ex
    io.shared.complete_st := io.vpu.complete_st
}

/**
  * Present one VPU dependency lifecycle to independent, private Gemmini ACC
  * owners.
  *
  * The VPU sees the private accumulators as one concatenated row address
  * space. Owner `i` occupies `[i * rowsPerOwner, (i + 1) * rowsPerOwner)`.
  * Each output mirrors the complete VPU lifecycle. An operand is intersected
  * with every owner region it overlaps, and each non-empty intersection is
  * rebased to zero. A vector command may consequently span several private
  * accumulators even though each physical memory request targets one owner.
  * This lets every private Gemmini retain a compact singleton
  * `FusionExtEntries`; no Gemmini-to-Gemmini table is introduced.
  *
  * Dependency readiness is the conjunction of all owner tables. Issue and
  * completion events are broadcast because every table allocates the same VPU
  * slot, including tables for which all operands are invalid.
  */
class PrivateAccVpuDepsFanout(
    localAddrType: LocalAddr,
    nOwners: Int,
    rowsPerOwner: Int,
    globalAddressBits: Int,
    reservationStationEntriesLd: Int,
    reservationStationEntriesEx: Int,
    reservationStationEntriesSt: Int,
    resMaxPerType: Int) extends Module {
    require(nOwners > 0)
    require(rowsPerOwner > 0)
    require(globalAddressBits >= log2Ceil(nOwners * rowsPerOwner + 1),
        "global VPU dependency address cannot represent every private ACC row")
    require(localAddrType.data.getWidth >= log2Ceil(rowsPerOwner),
        "Gemmini LocalAddr cannot represent one private ACC row space")

    val io = IO(new Bundle {
        val vpu = Flipped(new VpuEntriesForDeps(
            globalAddressBits,
            reservationStationEntriesLd,
            reservationStationEntriesEx,
            reservationStationEntriesSt,
            resMaxPerType))
        val owners = Vec(nOwners, new EntriesForDeps(
            localAddrType,
            reservationStationEntriesLd,
            reservationStationEntriesEx,
            reservationStationEntriesSt,
            resMaxPerType))
    })

    private val localDataBits = localAddrType.data.getWidth
    private val globalRows = nOwners * rowsPerOwner

    private def accAddress(localRow: UInt): LocalAddr = {
        val result = WireInit(0.U.asTypeOf(localAddrType))
        result.is_acc_addr := true.B
        result.data := localRow(localDataBits - 1, 0)
        result
    }

    private def overlapsOwner(in: VpuDepOperand, owner: Int): Bool = {
        val ownerStart = (owner * rowsPerOwner).U(globalAddressBits.W)
        val ownerEnd = ((owner + 1) * rowsPerOwner).U(globalAddressBits.W)
        in.start < ownerEnd && in.end > ownerStart && in.start < in.end
    }

    private def localize(
        in: VpuDepOperand,
        owner: Int): UDValid[OpT] = {
        val result = Wire(UDValid(new OpT(localAddrType)))
        val ownerStart = (owner * rowsPerOwner).U(globalAddressBits.W)
        val ownerEnd = ((owner + 1) * rowsPerOwner).U(globalAddressBits.W)
        val overlaps = overlapsOwner(in, owner)
        val clippedStart = Mux(in.start > ownerStart, in.start, ownerStart)
        val clippedEnd = Mux(in.end < ownerEnd, in.end, ownerEnd)
        val localStart = Mux(overlaps, clippedStart - ownerStart, 0.U)
        val localEnd = Mux(overlaps, clippedEnd - ownerStart, 0.U)

        result.valid := in.valid && overlaps
        result.bits.start := accAddress(localStart)
        result.bits.end := accAddress(localEnd)
        // The exclusive end of a complete owner region truncates to local row
        // zero in LocalAddr's ACC address field. Preserve OpT's wrap marker.
        result.bits.wraps_around :=
            in.valid && overlaps && clippedEnd === ownerEnd
        result
    }

    private val operands = Seq(
        io.vpu.alloc_entry.bits.opa,
        io.vpu.alloc_entry.bits.opb,
        io.vpu.alloc_entry.bits.opc)

    for (operand <- operands) {
        when (io.vpu.alloc_entry.valid && operand.valid) {
            assert(operand.start < operand.end,
                "private-ACC VPU dependency interval must be non-empty")
            assert(operand.end <= globalRows.U(globalAddressBits.W),
                "private-ACC VPU dependency interval escaped all owners")
        }
    }

    for ((owner, ownerId) <- io.owners.zipWithIndex) {
        owner.alloc_entry.valid := io.vpu.alloc_entry.valid
        owner.alloc_entry.bits.alloc_id := io.vpu.alloc_entry.bits.alloc_id
        owner.alloc_entry.bits.opa := localize(
            io.vpu.alloc_entry.bits.opa, ownerId)
        owner.alloc_entry.bits.opb := localize(
            io.vpu.alloc_entry.bits.opb, ownerId)
        owner.alloc_entry.bits.opc := localize(
            io.vpu.alloc_entry.bits.opc, ownerId)
        owner.alloc_entry.bits.opa_is_dst :=
            io.vpu.alloc_entry.bits.opa_is_dst
        owner.alloc_entry.bits.not_config := true.B

        owner.issue_ld := io.vpu.issue_ld
        owner.issue_ex := io.vpu.issue_ex
        owner.issue_st := io.vpu.issue_st
        owner.complete_ld := io.vpu.complete_ld
        owner.complete_ex := io.vpu.complete_ex
        owner.complete_st := io.vpu.complete_st
    }

    for (entry <- 0 until reservationStationEntriesLd) {
        io.vpu.ld_deps_ready(entry) :=
            io.owners.map(_.ld_deps_ready(entry)).reduce(_ && _)
    }
    for (entry <- 0 until reservationStationEntriesEx) {
        io.vpu.ex_deps_ready(entry) :=
            io.owners.map(_.ex_deps_ready(entry)).reduce(_ && _)
    }
    for (entry <- 0 until reservationStationEntriesSt) {
        io.vpu.st_deps_ready(entry) :=
            io.owners.map(_.st_deps_ready(entry)).reduce(_ && _)
    }
}

/**
  * Shared dependencies for all Gemmini owners and the optional VPU. The
  * fusion-predecessor organization is retained: exactly one LD array, one EX
  * array, and one ST array. VPU entries are appended slices of those arrays.
  */
class SharedExtEntries(
    nSharers: Int,
    local_addr_t: LocalAddr,
    reservation_station_entries_ld: Int,
    reservation_station_entries_ex: Int,
    reservation_station_entries_st: Int,
    res_max_per_type: Int,
    vpuEntriesLd: Int = 0,
    vpuEntriesEx: Int = 0,
    vpuEntriesSt: Int = 0) extends Module {

    require(nSharers > 0)
    require(vpuEntriesLd >= 0 && vpuEntriesEx >= 0 && vpuEntriesSt >= 0)
    private val hasVpu = vpuEntriesLd + vpuEntriesEx + vpuEntriesSt > 0

    private val gemEntriesLd = reservation_station_entries_ld * nSharers
    private val gemEntriesEx = reservation_station_entries_ex * nSharers
    private val gemEntriesSt = reservation_station_entries_st * nSharers
    private val totalEntriesLd = gemEntriesLd + vpuEntriesLd
    private val totalEntriesEx = gemEntriesEx + vpuEntriesEx
    private val totalEntriesSt = gemEntriesSt + vpuEntriesSt

    val io = IO(new Bundle {
        val in = Vec(nSharers, Flipped(new EntriesForDeps(
            local_addr_t,
            reservation_station_entries_ld,
            reservation_station_entries_ex,
            reservation_station_entries_st,
            res_max_per_type)))
        val vpu = if (hasVpu) Some(Flipped(new EntriesForDeps(
            local_addr_t,
            vpuEntriesLd,
            vpuEntriesEx,
            vpuEntriesSt,
            res_max_per_type))) else None
    })

    val ldq :: exq :: stq :: Nil = Enum(3)

    class Entry(
        dependencyEntries0: Int,
        dependencyEntries1: Int,
        trackVpuLd: Boolean = false) extends Bundle {
        val opa = UDValid(new OpT(local_addr_t))
        val opb = UDValid(new OpT(local_addr_t))
        val opc = UDValid(new OpT(local_addr_t))
        val opa_is_dst = Bool()

        val deps_0 = Vec(dependencyEntries0, Bool())
        val deps_1 = Vec(dependencyEntries1, Bool())
        val deps_vpu_ld = if (trackVpuLd) {
            Some(Vec(vpuEntriesLd, Bool()))
        } else None

        def ready(dummy: Int = 0): Bool = {
            val vpuLdBlocked = deps_vpu_ld
                .map(_.reduce(_ || _)).getOrElse(false.B)
            !(deps_0.reduce(_ || _) || deps_1.reduce(_ || _) ||
                vpuLdBlocked)
        }
    }

    val total_entries_ld = Reg(Vec(totalEntriesLd,
        UDValid(new Entry(totalEntriesEx, totalEntriesSt,
            trackVpuLd = vpuEntriesLd > 0))))
    val total_entries_ex = Reg(Vec(totalEntriesEx,
        UDValid(new Entry(totalEntriesLd, totalEntriesSt))))
    val total_entries_st = Reg(Vec(totalEntriesSt,
        UDValid(new Entry(totalEntriesLd, totalEntriesEx))))

    private val exOwners =
        (0 until nSharers).flatMap(owner =>
            Seq.fill(reservation_station_entries_ex)(owner)) ++
        Seq.fill(vpuEntriesEx)(nSharers)
    require(exOwners.size == totalEntriesEx)

    // Gemmini's local ReservationStation already tracks dependencies against
    // EX entries owned by the same Gemmini and releases them on issue. Store
    // only truly external EX blockers here: other Gemminis and the VPU. The
    // pipelined VPU still needs its own live EX entries in addition to all
    // Gemmini entries because its independent execution units can overlap.
    private val gemCrossExOldIndices = Seq.tabulate(nSharers) { owner =>
        (0 until totalEntriesEx).filter(exOwners(_) != owner)
    }
    private val gemCrossExDeps = gemCrossExOldIndices.map { oldIndices =>
        if (oldIndices.nonEmpty) {
            Some(Reg(Vec(reservation_station_entries_ex,
                Vec(oldIndices.size, Bool()))))
        } else None
    }
    private val vpuCrossExDeps = if (vpuEntriesEx > 0) {
        Some(Reg(Vec(vpuEntriesEx, Vec(totalEntriesEx, Bool()))))
    } else None

    private val typeWidth = log2Up(res_max_per_type)

    private def overlaps(a: UDValid[OpT], b: UDValid[OpT]): Bool =
        a.valid && b.valid && a.bits.overlaps(b.bits)

    /** EX/EX is tracked across owners and within the pipelined VPU. */
    private def exConflict(newEntry: AllocEntry, oldEntry: Entry): Bool = {
        val newWrite = newEntry.opa.valid && newEntry.opa_is_dst
        val oldWrite = oldEntry.opa.valid && oldEntry.opa_is_dst
        val newWriteOverlapsOld = newWrite && (
            overlaps(newEntry.opa, oldEntry.opa) ||
                overlaps(newEntry.opa, oldEntry.opb) ||
                overlaps(newEntry.opa, oldEntry.opc))
        val oldWriteOverlapsNew = oldWrite && (
            overlaps(oldEntry.opa, newEntry.opa) ||
                overlaps(oldEntry.opa, newEntry.opb) ||
                overlaps(oldEntry.opa, newEntry.opc))
        newEntry.not_config && (newWriteOverlapsOld || oldWriteOverlapsNew)
    }

    /** Pipelined VPU loads may retire out of order, so preserve LD/LD WAW. */
    private def ldConflict(newEntry: AllocEntry, oldEntry: Entry): Bool =
        newEntry.not_config && newEntry.opa_is_dst &&
            oldEntry.opa_is_dst && overlaps(newEntry.opa, oldEntry.opa)

    private def allocConflict(a: AllocEntry, b: AllocEntry): Bool = {
        val aWrite = a.opa.valid && a.opa_is_dst
        val bWrite = b.opa.valid && b.opa_is_dst
        val aWriteOverlapsB = aWrite && (
            overlaps(a.opa, b.opa) || overlaps(a.opa, b.opb) ||
                overlaps(a.opa, b.opc))
        val bWriteOverlapsA = bWrite && (
            overlaps(b.opa, a.opa) || overlaps(b.opa, a.opb) ||
                overlaps(b.opa, a.opc))
        a.not_config && b.not_config && (aWriteOverlapsB || bWriteOverlapsA)
    }

    private def allocate(
        client: EntriesForDeps,
        owner: Int,
        ldBase: Int,
        exBase: Int,
        stBase: Int,
        ldEntries: Int,
        exEntries: Int,
        stEntries: Int): Unit = {
        val allocation = client.alloc_entry
        val queue = allocation.bits.alloc_id(typeWidth + 1, typeWidth)
        val localId = allocation.bits.alloc_id(typeWidth - 1, 0)

        when(allocation.valid && queue === ldq) {
            assert(localId < ldEntries.U)
            val globalId = ldBase.U +& localId
            val target = total_entries_ld(globalId)
            assert(!target.valid,
                "shared dependency LD allocation overwrote a live slot")
            target.bits.deps_0 := VecInit(total_entries_ex.map { entry =>
                entry.valid && allocation.bits.not_config && (
                    overlaps(allocation.bits.opa, entry.bits.opa) ||
                    overlaps(allocation.bits.opa, entry.bits.opb) ||
                    overlaps(allocation.bits.opa, entry.bits.opc))
            })
            target.bits.deps_1 := VecInit(total_entries_st.map { entry =>
                entry.valid && allocation.bits.not_config &&
                    overlaps(allocation.bits.opa, entry.bits.opa)
            })
            target.bits.deps_vpu_ld.foreach { deps =>
                if (owner == nSharers) {
                    deps := VecInit(total_entries_ld
                        .slice(gemEntriesLd, totalEntriesLd).map { entry =>
                            entry.valid && ldConflict(
                                allocation.bits, entry.bits)
                        })
                } else {
                    deps := VecInit(Seq.fill(vpuEntriesLd)(false.B))
                }
            }
            target.valid := true.B
            target.bits.opa := allocation.bits.opa
            target.bits.opb := allocation.bits.opb
            target.bits.opc := allocation.bits.opc
            target.bits.opa_is_dst := allocation.bits.opa_is_dst
        }

        when(allocation.valid && queue === exq) {
            assert(localId < exEntries.U)
            val globalId = exBase.U +& localId
            val target = total_entries_ex(globalId)
            assert(!target.valid,
                "shared dependency EX allocation overwrote a live slot")
            target.bits.deps_0 := VecInit(total_entries_ld.map { entry =>
                entry.valid && allocation.bits.not_config && (
                    overlaps(allocation.bits.opa, entry.bits.opa) ||
                    overlaps(allocation.bits.opb, entry.bits.opa) ||
                    overlaps(allocation.bits.opc, entry.bits.opa))
            })
            target.bits.deps_1 := VecInit(total_entries_st.map { entry =>
                entry.valid && allocation.bits.not_config &&
                    allocation.bits.opa_is_dst &&
                    overlaps(allocation.bits.opa, entry.bits.opa)
            })
            if (owner < nSharers) {
                gemCrossExDeps(owner).foreach { deps =>
                    deps(localId) := VecInit(
                        gemCrossExOldIndices(owner).map { oldIndex =>
                            val entry = total_entries_ex(oldIndex)
                            entry.valid &&
                                exConflict(allocation.bits, entry.bits)
                        })
                }
            } else {
                vpuCrossExDeps.foreach { deps =>
                    deps(localId) := VecInit(total_entries_ex.map { entry =>
                        entry.valid &&
                            exConflict(allocation.bits, entry.bits)
                    })
                }
            }
            target.valid := true.B
            target.bits.opa := allocation.bits.opa
            target.bits.opb := allocation.bits.opb
            target.bits.opc := allocation.bits.opc
            target.bits.opa_is_dst := allocation.bits.opa_is_dst
        }

        when(allocation.valid && queue === stq) {
            assert(localId < stEntries.U)
            val globalId = stBase.U +& localId
            val target = total_entries_st(globalId)
            assert(!target.valid,
                "shared dependency ST allocation overwrote a live slot")
            target.bits.deps_0 := VecInit(total_entries_ld.map { entry =>
                entry.valid && allocation.bits.not_config &&
                    overlaps(allocation.bits.opa, entry.bits.opa)
            })
            target.bits.deps_1 := VecInit(total_entries_ex.map { entry =>
                entry.valid && entry.bits.opa_is_dst &&
                    allocation.bits.not_config &&
                    overlaps(allocation.bits.opa, entry.bits.opa)
            })
            target.valid := true.B
            target.bits.opa := allocation.bits.opa
            target.bits.opb := allocation.bits.opb
            target.bits.opc := allocation.bits.opc
            target.bits.opa_is_dst := allocation.bits.opa_is_dst
        }
    }

    for (owner <- 0 until nSharers) {
        allocate(io.in(owner), owner,
            ldBase = owner * reservation_station_entries_ld,
            exBase = owner * reservation_station_entries_ex,
            stBase = owner * reservation_station_entries_st,
            ldEntries = reservation_station_entries_ld,
            exEntries = reservation_station_entries_ex,
            stEntries = reservation_station_entries_st)
    }
    if (hasVpu) {
        allocate(io.vpu.get, nSharers,
            ldBase = gemEntriesLd,
            exBase = gemEntriesEx,
            stBase = gemEntriesSt,
            ldEntries = vpuEntriesLd,
            exEntries = vpuEntriesEx,
            stEntries = vpuEntriesSt)
    }

    // Valid-only allocation ports cannot establish age between simultaneous
    // conflicting allocations. Assert the unsupported case instead of adding
    // a same-cycle dependency path.
    for (first <- 0 until nSharers; second <- first + 1 until nSharers) {
        val a = io.in(first).alloc_entry
        val b = io.in(second).alloc_entry
        val aQueue = a.bits.alloc_id(typeWidth + 1, typeWidth)
        val bQueue = b.bits.alloc_id(typeWidth + 1, typeWidth)
        assert(!(a.valid && b.valid && aQueue === exq && bQueue === exq &&
            allocConflict(a.bits, b.bits)),
            "conflicting cross-Gemmini ACC EX allocations in one cycle")
    }
    if (hasVpu) {
        for (owner <- 0 until nSharers) {
            val a = io.in(owner).alloc_entry
            val b = io.vpu.get.alloc_entry
            assert(!(a.valid && b.valid && allocConflict(a.bits, b.bits)),
                "conflicting Gemmini/VPU ACC allocations in one cycle")
        }
    }

    for (owner <- 0 until nSharers) {
        io.in(owner).ld_deps_ready := total_entries_ld.slice(
            owner * reservation_station_entries_ld,
            (owner + 1) * reservation_station_entries_ld).map { entry =>
                !entry.valid || entry.bits.ready()
            }
        io.in(owner).ex_deps_ready := total_entries_ex.slice(
            owner * reservation_station_entries_ex,
            (owner + 1) * reservation_station_entries_ex)
            .zipWithIndex.map { case (entry, localId) =>
                val crossExReady = gemCrossExDeps(owner)
                    .map(deps => !deps(localId).asUInt.orR)
                    .getOrElse(true.B)
                !entry.valid || (entry.bits.ready() && crossExReady)
            }
        io.in(owner).st_deps_ready := total_entries_st.slice(
            owner * reservation_station_entries_st,
            (owner + 1) * reservation_station_entries_st).map { entry =>
                !entry.valid || entry.bits.ready()
            }
    }
    if (hasVpu) {
        io.vpu.get.ld_deps_ready := total_entries_ld
            .slice(gemEntriesLd, totalEntriesLd)
            .map(entry => !entry.valid || entry.bits.ready())
        io.vpu.get.ex_deps_ready := total_entries_ex
            .slice(gemEntriesEx, totalEntriesEx)
            .zipWithIndex.map { case (entry, localId) =>
                val crossExReady = vpuCrossExDeps
                    .map(deps => !deps(localId).asUInt.orR)
                    .getOrElse(true.B)
                !entry.valid || (entry.bits.ready() && crossExReady)
            }
        io.vpu.get.st_deps_ready := total_entries_st
            .slice(gemEntriesSt, totalEntriesSt)
            .map(entry => !entry.valid || entry.bits.ready())
    }

    private def clearLd(globalId: Int): Unit = {
        total_entries_ex.foreach(_.bits.deps_0(globalId) := false.B)
        total_entries_st.foreach(_.bits.deps_0(globalId) := false.B)
        if (globalId >= gemEntriesLd) {
            val vpuId = globalId - gemEntriesLd
            total_entries_ld.foreach(_.bits.deps_vpu_ld.foreach(
                _(vpuId) := false.B))
        }
        total_entries_ld(globalId).valid := false.B
    }

    private def clearEx(globalId: Int): Unit = {
        total_entries_ld.foreach(_.bits.deps_0(globalId) := false.B)
        total_entries_st.foreach(_.bits.deps_1(globalId) := false.B)
        for (owner <- 0 until nSharers) {
            val compressedIndex = gemCrossExOldIndices(owner)
                .indexOf(globalId)
            if (compressedIndex >= 0) {
                gemCrossExDeps(owner).foreach(
                    _.foreach(_(compressedIndex) := false.B))
            }
        }
        vpuCrossExDeps.foreach(_.foreach(_(globalId) := false.B))
        total_entries_ex(globalId).valid := false.B
    }

    private def clearSt(globalId: Int): Unit = {
        total_entries_ld.foreach(_.bits.deps_1(globalId) := false.B)
        total_entries_ex.foreach(_.bits.deps_1(globalId) := false.B)
        total_entries_st(globalId).valid := false.B
    }

    private def lifecycle(
        client: EntriesForDeps,
        ldBase: Int,
        exBase: Int,
        stBase: Int,
        ldEntries: Int,
        exEntries: Int,
        stEntries: Int): Unit = {
        when(client.issue_ld.valid) {
            val localId = client.issue_ld.bits.issue_id
            assert(localId < ldEntries.U)
            val globalId = ldBase.U +& localId
            total_entries_ld(globalId).valid := client.issue_ld.bits.valid
            when(!client.issue_ld.bits.valid) {
                for (index <- 0 until ldEntries) {
                    when(localId === index.U) { clearLd(ldBase + index) }
                }
            }
        }
        when(client.issue_ex.valid) {
            val localId = client.issue_ex.bits.issue_id
            assert(localId < exEntries.U)
            val globalId = exBase.U +& localId
            total_entries_ex(globalId).valid := client.issue_ex.bits.valid
            when(!client.issue_ex.bits.valid) {
                for (index <- 0 until exEntries) {
                    when(localId === index.U) { clearEx(exBase + index) }
                }
            }
        }
        when(client.issue_st.valid) {
            val localId = client.issue_st.bits.issue_id
            assert(localId < stEntries.U)
            val globalId = stBase.U +& localId
            total_entries_st(globalId).valid := client.issue_st.bits.valid
            when(!client.issue_st.bits.valid) {
                for (index <- 0 until stEntries) {
                    when(localId === index.U) { clearSt(stBase + index) }
                }
            }
        }

        for (index <- 0 until ldEntries) {
            when(client.complete_ld(index)) { clearLd(ldBase + index) }
        }
        for (index <- 0 until exEntries) {
            when(client.complete_ex(index)) { clearEx(exBase + index) }
        }
        for (index <- 0 until stEntries) {
            when(client.complete_st(index)) { clearSt(stBase + index) }
        }
    }

    for (owner <- 0 until nSharers) {
        lifecycle(io.in(owner),
            ldBase = owner * reservation_station_entries_ld,
            exBase = owner * reservation_station_entries_ex,
            stBase = owner * reservation_station_entries_st,
            ldEntries = reservation_station_entries_ld,
            exEntries = reservation_station_entries_ex,
            stEntries = reservation_station_entries_st)
    }
    if (hasVpu) {
        lifecycle(io.vpu.get,
            ldBase = gemEntriesLd,
            exBase = gemEntriesEx,
            stBase = gemEntriesSt,
            ldEntries = vpuEntriesLd,
            exEntries = vpuEntriesEx,
            stEntries = vpuEntriesSt)
    }

    Seq(total_entries_ld, total_entries_st).foreach { entries =>
        entries.foreach { entry =>
            entry.bits.opb.valid := false.B
            entry.bits.opb.bits := DontCare
            entry.bits.opc.valid := false.B
            entry.bits.opc.bits := DontCare
        }
    }

    when(reset.asBool) {
        total_entries_ld.foreach(_.valid := false.B)
        total_entries_ex.foreach(_.valid := false.B)
        total_entries_st.foreach(_.valid := false.B)
    }
}

/**
  * Fusion-only dependency table for a singleton Gemmini which keeps its
  * ordinary local ReservationStation dependencies.  Unlike SharedExtEntries,
  * this module stores no Gemmini-to-Gemmini dependency edges.  It contains
  * only Gemmini/VPU edges plus the VPU's cross-class and pipelined same-class
  * address edges. Gemmini same-owner ordering remains in its local station.
  *
  * The IO intentionally matches SharedExtEntries so the controller-side
  * wiring does not need a second lifecycle protocol.
  */
class FusionExtEntries(
    nSharers: Int,
    local_addr_t: LocalAddr,
    reservation_station_entries_ld: Int,
    reservation_station_entries_ex: Int,
    reservation_station_entries_st: Int,
    res_max_per_type: Int,
    vpuEntriesLd: Int,
    vpuEntriesEx: Int,
    vpuEntriesSt: Int) extends Module {

    require(nSharers == 1,
        "FusionExtEntries is only for a singleton, locally scheduled Gemmini")
    require(reservation_station_entries_ld > 0)
    require(reservation_station_entries_ex > 0)
    require(reservation_station_entries_st > 0)
    require(vpuEntriesLd > 0 && vpuEntriesEx > 0 && vpuEntriesSt > 0)

    private val gemLdEntries = reservation_station_entries_ld
    private val gemExEntries = reservation_station_entries_ex
    private val gemStEntries = reservation_station_entries_st
    private val typeWidth = log2Up(res_max_per_type)

    val io = IO(new Bundle {
        val in = Vec(nSharers, Flipped(new EntriesForDeps(
            local_addr_t,
            reservation_station_entries_ld,
            reservation_station_entries_ex,
            reservation_station_entries_st,
            res_max_per_type)))
        val vpu = Some(Flipped(new EntriesForDeps(
            local_addr_t,
            vpuEntriesLd,
            vpuEntriesEx,
            vpuEntriesSt,
            res_max_per_type)))
    })

    val ldq :: exq :: stq :: Nil = Enum(3)

    class SingleAccess extends Bundle {
        val opa = UDValid(new OpT(local_addr_t))
    }

    class ExecuteAccess extends Bundle {
        val opa = UDValid(new OpT(local_addr_t))
        val opb = UDValid(new OpT(local_addr_t))
        val opc = UDValid(new OpT(local_addr_t))
        val opa_is_dst = Bool()
    }

    // Access descriptors are needed only for future cross-owner comparisons.
    val gemLd = Reg(Vec(gemLdEntries, UDValid(new SingleAccess)))
    val gemEx = Reg(Vec(gemExEntries, UDValid(new ExecuteAccess)))
    val gemSt = Reg(Vec(gemStEntries, UDValid(new SingleAccess)))
    val vpuLd = Reg(Vec(vpuEntriesLd, UDValid(new SingleAccess)))
    val vpuEx = Reg(Vec(vpuEntriesEx, UDValid(new ExecuteAccess)))
    val vpuSt = Reg(Vec(vpuEntriesSt, UDValid(new SingleAccess)))

    // Each vector contains only blockers which are external to the target's
    // own local dependency table. Segment layouts are documented at each use.
    val gemLdDeps = Reg(Vec(gemLdEntries,
        Vec(vpuEntriesEx + vpuEntriesSt, Bool()))) // VEX | VST
    val gemExDeps = Reg(Vec(gemExEntries,
        Vec(vpuEntriesLd + vpuEntriesSt + vpuEntriesEx, Bool())))
        // VLD | VST | VEX
    val gemStDeps = Reg(Vec(gemStEntries,
        Vec(vpuEntriesLd + vpuEntriesEx, Bool()))) // VLD | VEX

    val vpuLdDeps = Reg(Vec(vpuEntriesLd,
        Vec(gemExEntries + gemStEntries + vpuEntriesEx + vpuEntriesSt +
            vpuEntriesLd, Bool()))) // GEX | GST | VEX | VST | VLD
    val vpuExDeps = Reg(Vec(vpuEntriesEx,
        Vec(gemLdEntries + gemStEntries + gemExEntries + vpuEntriesLd +
            vpuEntriesSt + vpuEntriesEx, Bool())))
        // GLD | GST | GEX | VLD | VST | VEX
    val vpuStDeps = Reg(Vec(vpuEntriesSt,
        Vec(gemLdEntries + gemExEntries + vpuEntriesLd + vpuEntriesEx,
            Bool()))) // GLD | GEX | VLD | VEX

    private def overlaps(a: UDValid[OpT], b: UDValid[OpT]): Bool =
        a.valid && b.valid && a.bits.overlaps(b.bits)

    private def exConflict(
        newEntry: AllocEntry,
        oldEntry: ExecuteAccess): Bool = {
        val newWrite = newEntry.opa.valid && newEntry.opa_is_dst
        val oldWrite = oldEntry.opa.valid && oldEntry.opa_is_dst
        val newWriteOverlapsOld = newWrite && (
            overlaps(newEntry.opa, oldEntry.opa) ||
                overlaps(newEntry.opa, oldEntry.opb) ||
                overlaps(newEntry.opa, oldEntry.opc))
        val oldWriteOverlapsNew = oldWrite && (
            overlaps(oldEntry.opa, newEntry.opa) ||
                overlaps(oldEntry.opa, newEntry.opb) ||
                overlaps(oldEntry.opa, newEntry.opc))
        newEntry.not_config && (newWriteOverlapsOld || oldWriteOverlapsNew)
    }

    private def allocConflict(a: AllocEntry, b: AllocEntry): Bool = {
        val aWrite = a.opa.valid && a.opa_is_dst
        val bWrite = b.opa.valid && b.opa_is_dst
        val aWriteOverlapsB = aWrite && (
            overlaps(a.opa, b.opa) || overlaps(a.opa, b.opb) ||
                overlaps(a.opa, b.opc))
        val bWriteOverlapsA = bWrite && (
            overlaps(b.opa, a.opa) || overlaps(b.opa, a.opb) ||
                overlaps(b.opa, a.opc))
        a.not_config && b.not_config && (aWriteOverlapsB || bWriteOverlapsA)
    }

    private def recordSingle(
        target: UDValid[SingleAccess],
        allocation: AllocEntry): Unit = {
        target.valid := true.B
        target.bits.opa := allocation.opa
    }

    private def recordExecute(
        target: UDValid[ExecuteAccess],
        allocation: AllocEntry): Unit = {
        target.valid := true.B
        target.bits.opa := allocation.opa
        target.bits.opb := allocation.opb
        target.bits.opc := allocation.opc
        target.bits.opa_is_dst := allocation.opa_is_dst
    }

    private def ldAfterEx(
        allocation: AllocEntry,
        old: UDValid[ExecuteAccess]): Bool =
        old.valid && allocation.not_config && (
            overlaps(allocation.opa, old.bits.opa) ||
                overlaps(allocation.opa, old.bits.opb) ||
                overlaps(allocation.opa, old.bits.opc))

    private def ldAfterSt(
        allocation: AllocEntry,
        old: UDValid[SingleAccess]): Bool =
        old.valid && allocation.not_config &&
            overlaps(allocation.opa, old.bits.opa)

    private def ldAfterLd(
        allocation: AllocEntry,
        old: UDValid[SingleAccess]): Bool =
        old.valid && allocation.not_config && allocation.opa_is_dst &&
            overlaps(allocation.opa, old.bits.opa)

    private def exAfterLd(
        allocation: AllocEntry,
        old: UDValid[SingleAccess]): Bool =
        old.valid && allocation.not_config && (
            overlaps(allocation.opa, old.bits.opa) ||
                overlaps(allocation.opb, old.bits.opa) ||
                overlaps(allocation.opc, old.bits.opa))

    private def exAfterSt(
        allocation: AllocEntry,
        old: UDValid[SingleAccess]): Bool =
        old.valid && allocation.not_config && allocation.opa_is_dst &&
            overlaps(allocation.opa, old.bits.opa)

    private def stAfterLd(
        allocation: AllocEntry,
        old: UDValid[SingleAccess]): Bool =
        old.valid && allocation.not_config &&
            overlaps(allocation.opa, old.bits.opa)

    private def stAfterEx(
        allocation: AllocEntry,
        old: UDValid[ExecuteAccess]): Bool =
        old.valid && old.bits.opa_is_dst && allocation.not_config &&
            overlaps(allocation.opa, old.bits.opa)

    private def allocateGemmini(client: EntriesForDeps): Unit = {
        val allocation = client.alloc_entry
        val queue = allocation.bits.alloc_id(typeWidth + 1, typeWidth)
        val localId = allocation.bits.alloc_id(typeWidth - 1, 0)

        when(allocation.valid && queue === ldq) {
            assert(localId < gemLdEntries.U)
            assert(!gemLd(localId).valid,
                "fusion dependency Gemmini LD allocation overwrote a live slot")
            recordSingle(gemLd(localId), allocation.bits)
            gemLdDeps(localId) := VecInit(
                vpuEx.map(ldAfterEx(allocation.bits, _)) ++
                    vpuSt.map(ldAfterSt(allocation.bits, _)))
        }
        when(allocation.valid && queue === exq) {
            assert(localId < gemExEntries.U)
            assert(!gemEx(localId).valid,
                "fusion dependency Gemmini EX allocation overwrote a live slot")
            recordExecute(gemEx(localId), allocation.bits)
            gemExDeps(localId) := VecInit(
                vpuLd.map(exAfterLd(allocation.bits, _)) ++
                    vpuSt.map(exAfterSt(allocation.bits, _)) ++
                    vpuEx.map(entry => entry.valid &&
                        exConflict(allocation.bits, entry.bits)))
        }
        when(allocation.valid && queue === stq) {
            assert(localId < gemStEntries.U)
            assert(!gemSt(localId).valid,
                "fusion dependency Gemmini ST allocation overwrote a live slot")
            recordSingle(gemSt(localId), allocation.bits)
            gemStDeps(localId) := VecInit(
                vpuLd.map(stAfterLd(allocation.bits, _)) ++
                    vpuEx.map(stAfterEx(allocation.bits, _)))
        }
    }

    private def allocateVpu(client: EntriesForDeps): Unit = {
        val allocation = client.alloc_entry
        val queue = allocation.bits.alloc_id(typeWidth + 1, typeWidth)
        val localId = allocation.bits.alloc_id(typeWidth - 1, 0)

        when(allocation.valid && queue === ldq) {
            assert(localId < vpuEntriesLd.U)
            assert(!vpuLd(localId).valid,
                "fusion dependency VPU LD allocation overwrote a live slot")
            recordSingle(vpuLd(localId), allocation.bits)
            vpuLdDeps(localId) := VecInit(
                gemEx.map(ldAfterEx(allocation.bits, _)) ++
                    gemSt.map(ldAfterSt(allocation.bits, _)) ++
                    vpuEx.map(ldAfterEx(allocation.bits, _)) ++
                    vpuSt.map(ldAfterSt(allocation.bits, _)) ++
                    vpuLd.map(ldAfterLd(allocation.bits, _)))
        }
        when(allocation.valid && queue === exq) {
            assert(localId < vpuEntriesEx.U)
            assert(!vpuEx(localId).valid,
                "fusion dependency VPU EX allocation overwrote a live slot")
            recordExecute(vpuEx(localId), allocation.bits)
            vpuExDeps(localId) := VecInit(
                gemLd.map(exAfterLd(allocation.bits, _)) ++
                    gemSt.map(exAfterSt(allocation.bits, _)) ++
                    gemEx.map(entry => entry.valid &&
                        exConflict(allocation.bits, entry.bits)) ++
                    vpuLd.map(exAfterLd(allocation.bits, _)) ++
                    vpuSt.map(exAfterSt(allocation.bits, _)) ++
                    vpuEx.map(entry => entry.valid &&
                        exConflict(allocation.bits, entry.bits)))
        }
        when(allocation.valid && queue === stq) {
            assert(localId < vpuEntriesSt.U)
            assert(!vpuSt(localId).valid,
                "fusion dependency VPU ST allocation overwrote a live slot")
            recordSingle(vpuSt(localId), allocation.bits)
            vpuStDeps(localId) := VecInit(
                gemLd.map(stAfterLd(allocation.bits, _)) ++
                    gemEx.map(stAfterEx(allocation.bits, _)) ++
                    vpuLd.map(stAfterLd(allocation.bits, _)) ++
                    vpuEx.map(stAfterEx(allocation.bits, _)))
        }
    }

    allocateGemmini(io.in.head)
    allocateVpu(io.vpu.get)

    // Valid-only allocation ports do not carry age between simultaneous
    // owners. Preserve the shared-table rule: reject a conflicting pair.
    assert(!(io.in.head.alloc_entry.valid && io.vpu.get.alloc_entry.valid &&
        allocConflict(io.in.head.alloc_entry.bits,
            io.vpu.get.alloc_entry.bits)),
        "conflicting Gemmini/VPU ACC allocations in one cycle")

    io.in.head.ld_deps_ready := gemLd.zip(gemLdDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }
    io.in.head.ex_deps_ready := gemEx.zip(gemExDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }
    io.in.head.st_deps_ready := gemSt.zip(gemStDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }
    io.vpu.get.ld_deps_ready := vpuLd.zip(vpuLdDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }
    io.vpu.get.ex_deps_ready := vpuEx.zip(vpuExDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }
    io.vpu.get.st_deps_ready := vpuSt.zip(vpuStDeps).map {
        case (entry, deps) => !entry.valid || !deps.asUInt.orR
    }

    private def clearGemLd(index: Int): Unit = {
        vpuExDeps.foreach(_(index) := false.B)
        vpuStDeps.foreach(_(index) := false.B)
        gemLd(index).valid := false.B
    }

    private def clearGemEx(index: Int): Unit = {
        vpuLdDeps.foreach(_(index) := false.B)
        vpuExDeps.foreach(_(gemLdEntries + gemStEntries + index) := false.B)
        vpuStDeps.foreach(_(gemLdEntries + index) := false.B)
        gemEx(index).valid := false.B
    }

    private def clearGemSt(index: Int): Unit = {
        vpuLdDeps.foreach(_(gemExEntries + index) := false.B)
        vpuExDeps.foreach(_(gemLdEntries + index) := false.B)
        gemSt(index).valid := false.B
    }

    private def clearVpuLd(index: Int): Unit = {
        gemExDeps.foreach(_(index) := false.B)
        gemStDeps.foreach(_(index) := false.B)
        vpuExDeps.foreach(_(
            gemLdEntries + gemStEntries + gemExEntries + index) := false.B)
        vpuStDeps.foreach(_(gemLdEntries + gemExEntries + index) := false.B)
        vpuLdDeps.foreach(_(
            gemExEntries + gemStEntries + vpuEntriesEx + vpuEntriesSt +
                index) := false.B)
        vpuLd(index).valid := false.B
    }

    private def clearVpuEx(index: Int): Unit = {
        gemLdDeps.foreach(_(index) := false.B)
        gemExDeps.foreach(_(vpuEntriesLd + vpuEntriesSt + index) := false.B)
        gemStDeps.foreach(_(vpuEntriesLd + index) := false.B)
        vpuLdDeps.foreach(_(gemExEntries + gemStEntries + index) := false.B)
        vpuStDeps.foreach(_(
            gemLdEntries + gemExEntries + vpuEntriesLd + index) := false.B)
        vpuExDeps.foreach(_(
            gemLdEntries + gemStEntries + gemExEntries + vpuEntriesLd +
                vpuEntriesSt + index) := false.B)
        vpuEx(index).valid := false.B
    }

    private def clearVpuSt(index: Int): Unit = {
        gemLdDeps.foreach(_(vpuEntriesEx + index) := false.B)
        gemExDeps.foreach(_(vpuEntriesLd + index) := false.B)
        vpuLdDeps.foreach(_(
            gemExEntries + gemStEntries + vpuEntriesEx + index) := false.B)
        vpuExDeps.foreach(_(
            gemLdEntries + gemStEntries + gemExEntries + vpuEntriesLd +
                index) := false.B)
        vpuSt(index).valid := false.B
    }

    private def lifecycleGemmini(client: EntriesForDeps): Unit = {
        when(client.issue_ld.valid) {
            val localId = client.issue_ld.bits.issue_id
            assert(localId < gemLdEntries.U)
            gemLd(localId).valid := client.issue_ld.bits.valid
            when(!client.issue_ld.bits.valid) {
                for (index <- 0 until gemLdEntries) {
                    when(localId === index.U) { clearGemLd(index) }
                }
            }
        }
        when(client.issue_ex.valid) {
            val localId = client.issue_ex.bits.issue_id
            assert(localId < gemExEntries.U)
            gemEx(localId).valid := client.issue_ex.bits.valid
            when(!client.issue_ex.bits.valid) {
                for (index <- 0 until gemExEntries) {
                    when(localId === index.U) { clearGemEx(index) }
                }
            }
        }
        when(client.issue_st.valid) {
            val localId = client.issue_st.bits.issue_id
            assert(localId < gemStEntries.U)
            gemSt(localId).valid := client.issue_st.bits.valid
            when(!client.issue_st.bits.valid) {
                for (index <- 0 until gemStEntries) {
                    when(localId === index.U) { clearGemSt(index) }
                }
            }
        }
        for (index <- 0 until gemLdEntries) {
            when(client.complete_ld(index)) { clearGemLd(index) }
        }
        for (index <- 0 until gemExEntries) {
            when(client.complete_ex(index)) { clearGemEx(index) }
        }
        for (index <- 0 until gemStEntries) {
            when(client.complete_st(index)) { clearGemSt(index) }
        }
    }

    private def lifecycleVpu(client: EntriesForDeps): Unit = {
        when(client.issue_ld.valid) {
            val localId = client.issue_ld.bits.issue_id
            assert(localId < vpuEntriesLd.U)
            vpuLd(localId).valid := client.issue_ld.bits.valid
            when(!client.issue_ld.bits.valid) {
                for (index <- 0 until vpuEntriesLd) {
                    when(localId === index.U) { clearVpuLd(index) }
                }
            }
        }
        when(client.issue_ex.valid) {
            val localId = client.issue_ex.bits.issue_id
            assert(localId < vpuEntriesEx.U)
            vpuEx(localId).valid := client.issue_ex.bits.valid
            when(!client.issue_ex.bits.valid) {
                for (index <- 0 until vpuEntriesEx) {
                    when(localId === index.U) { clearVpuEx(index) }
                }
            }
        }
        when(client.issue_st.valid) {
            val localId = client.issue_st.bits.issue_id
            assert(localId < vpuEntriesSt.U)
            vpuSt(localId).valid := client.issue_st.bits.valid
            when(!client.issue_st.bits.valid) {
                for (index <- 0 until vpuEntriesSt) {
                    when(localId === index.U) { clearVpuSt(index) }
                }
            }
        }
        for (index <- 0 until vpuEntriesLd) {
            when(client.complete_ld(index)) { clearVpuLd(index) }
        }
        for (index <- 0 until vpuEntriesEx) {
            when(client.complete_ex(index)) { clearVpuEx(index) }
        }
        for (index <- 0 until vpuEntriesSt) {
            when(client.complete_st(index)) { clearVpuSt(index) }
        }
    }

    lifecycleGemmini(io.in.head)
    lifecycleVpu(io.vpu.get)

    when(reset.asBool) {
        gemLd.foreach(_.valid := false.B)
        gemEx.foreach(_.valid := false.B)
        gemSt.foreach(_.valid := false.B)
        vpuLd.foreach(_.valid := false.B)
        vpuEx.foreach(_.valid := false.B)
        vpuSt.foreach(_.valid := false.B)
    }
}
