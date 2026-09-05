package gemmini

import chisel3._
import chisel3.util._

/**
  * Canonical progress of the load stream cooperatively produced by a group.
  *
  * `next_outer` and `next_inner` identify the next command not yet registered,
  * independent of the physical/transposed DMA layout. The owner range is the
  * logical coordinate interval assigned to this member:
  *   - M: LdB, outer=k, inner=j, owner range=K
  *   - N: LdA, outer=k, inner=i, owner range=I
  *   - K: LdB, outer=k, inner=j, owner range=K (admission only)
  */
class CoopLoadState(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int,
  useVpuFusion: Boolean
) extends Bundle {
  val axis = UInt(GemminiISA.PartitionAxis.width.W)
  val group_id = UInt(group_w.W)
  val member_mask = UInt(nSharers.W)
  val wait_event_valid = if (useVpuFusion) Some(Bool()) else None
  val wait_event_id = if (useVpuFusion) Some(UInt(EventTracker.IdWidth.W)) else None
  val produce_event_valid = if (useVpuFusion) Some(Bool()) else None
  val produce_event_id = if (useVpuFusion) Some(UInt(EventTracker.IdWidth.W)) else None
  val produce_event_seal = if (useVpuFusion) Some(Bool()) else None
  val next_outer = UInt(iterator_bitwidth.W)
  val next_inner = UInt(iterator_bitwidth.W)
  val owner_offset = UInt(iterator_bitwidth.W)
  val owner_extent = UInt(iterator_bitwidth.W)
  val idle = Bool()
}

class ExState(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val axis = UInt(GemminiISA.PartitionAxis.width.W)
  val group_id = UInt(group_w.W)
  val member_mask = UInt(nSharers.W)
  val k = UInt(iterator_bitwidth.W)
  val j = UInt(iterator_bitwidth.W)
  val i = UInt(iterator_bitwidth.W)
  val is_first_k_shard = Bool()
  val is_compute = Bool()
  val registration_fire = Bool()
  val is_last_local_k = Bool()
  // True only when this K group starts a fresh, bias-free accumulator. The
  // K-offset-zero member overwrites each C tile on local k=0; every other K
  // writer waits for that overwrite to be registered and then accumulates.
  val first_k_overwrites = Bool()
  val idle = Bool()
}

class LdDState(
  group_w: Int
) extends Bundle {
  val axis = UInt(GemminiISA.PartitionAxis.width.W)
  val group_id = UInt(group_w.W)
  val idle = Bool()
}

class StCState(
  group_w: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val axis = UInt(GemminiISA.PartitionAxis.width.W)
  val group_id = UInt(group_w.W)
  val global_j = UInt(iterator_bitwidth.W)
  val global_i = UInt(iterator_bitwidth.W)
  val j_blocks = UInt(iterator_bitwidth.W)
  // Request excludes the K group grant, so the group controller can
  // arbitrate without creating a combinational valid/grant loop.
  val registration_request = Bool()
  val registration_fire = Bool()
  val idle = Bool()
}

class LoopMatmulGroupIO(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int,
  useVpuFusion: Boolean
) extends Bundle {
  val lda = Output(new CoopLoadState(group_w, nSharers, iterator_bitwidth, useVpuFusion))
  val ldb = Output(new CoopLoadState(group_w, nSharers, iterator_bitwidth, useVpuFusion))
  val ldd = Output(new LdDState(group_w))
  val ex = Output(new ExState(group_w, nSharers, iterator_bitwidth))
  val stc = Output(new StCState(group_w, iterator_bitwidth))
  val coop_operand_registered = Input(Bool())
  // LdA and LdB can belong to different entries in LoopMatmul's two-loop
  // window. Keep their admission backpressure independent.
  val lda_admission_blocked = Input(Bool())
  val ldb_admission_blocked = Input(Bool())
  // K-only registration and same-tile pair gate. True for M/N.
  val k_execute_ready = Input(Bool())
  val k_store_grant = Input(Bool())
}

/**
  * Group-level handshake between the multi-Gemmini completion controller and
  * the VPU command gate. The VPU is deliberately not modelled as another
  * Gemmini sharer: it waits until every Gemmini child has been atomically
  * allocated into the Gemmini reservation stations/shared dependency table,
  * then allocates its commands into the corresponding VPU structures. Actual
  * functional-unit completion is deliberately not a dispatch prerequisite;
  * SharedExtEntries orders conflicting accesses until completion.
  *
  * `last_dispatch_fire` must only pulse when the final grouped VPU command has
  * actually been accepted by both of those structures. Merely buffering the
  * command in the VPU's one-entry group gate is not sufficient.
  */
class GemvGroupIO(group_w: Int) extends Bundle {
  val pending = Input(Bool())
  val group_id = Input(UInt(group_w.W))

  val group_allocated = Output(Bool())
  val dispatch_enable = Output(Bool())
  /** The queried allocated group did not request a VPU follow-up. */
  val dispatch_reject = Output(Bool())

  val last_dispatch_fire = Input(Bool())
  val abort = Input(Bool())
}

/** Minimal group-admission state used when VPU fusion is enabled without the
  * shared/MNK LOOP_WS scheduler.  LdB remains the legacy M-loop admission
  * anchor, but no partition axis, owner range, or iterator progress crosses
  * this interface.
  */
class FusionLdbGroupState(
  group_w: Int,
  nSharers: Int
) extends Bundle {
  val group_id = UInt(group_w.W)
  val member_mask = UInt(nSharers.W)
  val has_gemv_followup = Bool()
  val idle = Bool()
}

/** Fusion-only group lifecycle interface.  `completion` pulses after all five
  * LoopMatmul child streams (LdA/LdB/LdD/Execute/Store) for the local loop have
  * finished registering their commands.  Functional-unit completion remains
  * ordered by the reservation dependencies and is intentionally not tracked
  * here.
  */
class FusionLoopMatmulGroupIO(
  group_w: Int,
  nSharers: Int
) extends Bundle {
  val ldb = Output(new FusionLdbGroupState(group_w, nSharers))
  val completion = Output(Valid(UInt(group_w.W)))
  val ldb_admission_blocked = Input(Bool())
}

/**
  * Group lifecycle needed by fusion when shared/MNK scheduling is disabled.
  *
  * The finite group ID is the physical table index.  The first active LdB
  * anchor allocates the slot; the member mask names every Gemmini which must
  * subsequently report completion.  `ldb_done` prevents an early command that
  * reuses a still-live group ID from being mistaken for the old member's open
  * LdB stream.  No MNK axis, cooperative data-ahead, K overwrite frontier, or
  * K Store arbitration is elaborated in this module.
  */
class FusionLoopMatmulGroupController(
  nSharers: Int
) extends Module {
  private val groupNum = 8
  private val groupW = 3

  require(nSharers > 0)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new FusionLoopMatmulGroupIO(
      groupW, nSharers)))
    val gemv = new GemvGroupIO(groupW)
  })

  class GroupData extends Bundle {
    val member_mask = UInt(nSharers.W)
    val ldb_done = Vec(nSharers, Bool())
    val member_registered = Vec(nSharers, Bool())
    val gemv_required = Bool()
  }

  val groupData = Reg(Vec(groupNum, Valid(new GroupData)))

  def validGroupId(groupId: UInt): Bool = groupId < groupNum.U
  def allMembersRegistered(group: GroupData): Bool =
    group.member_registered.reduce(_ && _)

  val ldbIdleDelayed = RegNext(
    VecInit(io.in.map(_.ldb.idle)), VecInit(Seq.fill(nSharers)(true.B)))

  // A member may emit the current LdB stream only after its group has been
  // allocated and until that stream has closed.  A later loop reusing the same
  // live group ID therefore remains blocked.
  for (member <- 0 until nSharers) {
    val ldb = io.in(member).ldb
    val validId = validGroupId(ldb.group_id)
    val group = groupData(ldb.group_id)
    val ongoing = validId && group.valid &&
      group.bits.member_mask(member) && !group.bits.ldb_done(member)
    io.in(member).ldb_admission_blocked := !ldb.idle && !ongoing
  }

  // The group ID itself selects the slot.  Priority is relevant only when
  // multiple members present the same new group in one cycle; their metadata
  // must be identical, so any one of them can initialize the table entry.
  for (groupId <- 0 until groupNum) {
    val group = groupData(groupId)
    val candidates = VecInit(io.in.map(in =>
      !in.ldb.idle && validGroupId(in.ldb.group_id) &&
        in.ldb.group_id === groupId.U && !group.valid))
    val winnerOH = PriorityEncoderOH(candidates.asUInt)
    val winner = Mux1H(winnerOH, io.in.map(_.ldb))
    val canAllocate = candidates.asUInt.orR

    when (canAllocate) {
      group.valid := true.B
      group.bits.member_mask := winner.member_mask
      group.bits.gemv_required := winner.has_gemv_followup
      for (member <- 0 until nSharers) {
        group.bits.ldb_done(member) := !winner.member_mask(member)
        group.bits.member_registered(member) := !winner.member_mask(member)
      }
    }

    for (member <- 0 until nSharers) {
      when (candidates(member)) {
        assert(io.in(member).ldb.member_mask === winner.member_mask,
          "fusion group members disagreed on member_mask")
        assert(io.in(member).ldb.has_gemv_followup ===
          winner.has_gemv_followup,
          "fusion group members disagreed on has_gemv_followup")
      }
    }
  }

  // Record closure of the admission anchor.  This is separate from complete
  // child registration because Execute/Store may continue after LdB drains.
  for (member <- 0 until nSharers) {
    val ldb = io.in(member).ldb
    when (ldb.idle && !ldbIdleDelayed(member)) {
      assert(validGroupId(ldb.group_id),
        "fusion LdB completion carried an invalid group ID")
      val group = groupData(ldb.group_id)
      assert(group.valid && group.bits.member_mask(member),
        "fusion LdB completion targeted an unallocated group/member")
      group.bits.ldb_done(member) := true.B
    }

    when (io.in(member).completion.valid) {
      val groupId = io.in(member).completion.bits
      assert(validGroupId(groupId),
        "fusion LoopMatmul completion carried an invalid group ID")
      val group = groupData(groupId)
      assert(group.valid && group.bits.member_mask(member),
        "fusion LoopMatmul completion targeted an unallocated group/member")
      group.bits.member_registered(member) := true.B
    }
  }

  val gemvGroup = groupData(io.gemv.group_id)
  val gemvGroupAllocated = validGroupId(io.gemv.group_id) && gemvGroup.valid
  val gemvMembersRegistered = gemvGroupAllocated &&
    allMembersRegistered(gemvGroup.bits)
  val gemvRequired = gemvGroupAllocated && gemvGroup.bits.gemv_required

  io.gemv.group_allocated := gemvGroupAllocated
  io.gemv.dispatch_enable := gemvGroupAllocated &&
    gemvMembersRegistered && gemvRequired
  io.gemv.dispatch_reject := gemvGroupAllocated && !gemvRequired

  when (io.gemv.last_dispatch_fire) {
    assert(io.gemv.pending,
      "last_dispatch_fire requires a pending grouped VPU command")
    assert(io.gemv.dispatch_enable,
      "last grouped VPU command fired before Gemmini child registration")
  }
  when (io.gemv.abort) {
    assert(io.gemv.pending,
      "VPU group abort requires a pending grouped VPU command")
    assert(gemvGroupAllocated,
      "VPU group abort targeted an unallocated group")
    when (gemvRequired) {
      assert(gemvMembersRegistered,
        "VPU group aborted before all Gemmini children were registered")
    }
  }

  groupData.zipWithIndex.foreach { case (group, groupId) =>
    val vpuClosesGroup =
      (io.gemv.last_dispatch_fire || io.gemv.abort) &&
        io.gemv.group_id === groupId.U
    val groupDone = Mux(group.bits.gemv_required, vpuClosesGroup,
      allMembersRegistered(group.bits))
    when (group.valid && groupDone) {
      group.valid := false.B
    }
  }

  when (reset.asBool) {
    groupData.foreach(_.valid := false.B)
  }
}

class LoopMatmulGroupController(
  nSharers: Int,
  useVpuFusion: Boolean = false
) extends Module {
  val iterator_bitwidth = 16
  val concurrent_loops = 2
  val group_num = nSharers * concurrent_loops
  val group_w = log2Up(group_num)
  val member_w = log2Ceil(nSharers max 2)

  require(nSharers > 0)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new LoopMatmulGroupIO(
      group_w, nSharers, iterator_bitwidth, useVpuFusion)))
    val event_admissions = if (useVpuFusion) {
      Some(Vec(group_num, Flipped(new EventTrackerAdmissionPort)))
    } else None
    val event_completions = if (useVpuFusion) {
      Some(Vec(group_num, Flipped(new EventTrackerCompletionPort)))
    } else None
  })

  io.in.foreach(_.coop_operand_registered := false.B)
  io.in.foreach(_.k_execute_ready := true.B)
  io.in.foreach(_.k_store_grant := true.B)

  class LoadOwnerRange extends Bundle {
    val offset = UInt(iterator_bitwidth.W)
    val extent = UInt(iterator_bitwidth.W)
  }

  class TileCoord extends Bundle {
    val j = UInt(iterator_bitwidth.W)
    val i = UInt(iterator_bitwidth.W)
  }

  class MemberData extends Bundle {
    val completed_load_range = Valid(new LoadOwnerRange)
    val d_load_stream_registered = Bool()
    val execute_stream_registered = Bool()
    val store_stream_registered = Bool()
  }

  class GroupData extends Bundle {
    val mem_data = Vec(nSharers, new MemberData)
    val axis = UInt(GemminiISA.PartitionAxis.width.W)
    val member_mask = UInt(nSharers.W)
    // Highest tile whose offset-zero/local-k-zero overwrite COMPUTE has been
    // registered. This is a registration watermark, not FU completion.
    val k_overwrite_frontier = Valid(new TileCoord)
    val k_store_rr_ptr = UInt(member_w.W)
    val produce_event_valid = if (useVpuFusion) Some(Bool()) else None
    val produce_event_id = if (useVpuFusion) Some(UInt(EventTracker.IdWidth.W)) else None
  }

  val group_data = Reg(Vec(group_num, Valid(new GroupData)))
  // Execute can only drain after its local operand loads and the selected
  // cooperative stream are ahead. Consequently these completion bits cover
  // registration of the complete Gemmini child stream; no FU-completion latch
  // is needed here.
  def allMemberStreamsRegistered(memData: Vec[MemberData]): Bool =
    memData.map(md =>
      md.completed_load_range.valid && md.d_load_stream_registered &&
        md.execute_stream_registered && md.store_stream_registered
    ).reduce(_ && _)

  def tileEq(aJ: UInt, aI: UInt, bJ: UInt, bI: UInt): Bool =
    aJ === bJ && aI === bI

  def tileGt(aJ: UInt, aI: UInt, bJ: UInt, bI: UInt): Bool =
    aJ > bJ || (aJ === bJ && aI > bI)

  def tileGe(aJ: UInt, aI: UInt, bJ: UInt, bI: UInt): Bool =
    tileGt(aJ, aI, bJ, bI) || tileEq(aJ, aI, bJ, bI)

  def isEmptyKShard(member: MemberData): Bool =
    member.completed_load_range.valid &&
      member.completed_load_range.bits.extent === 0.U

  def validGroupId(groupId: UInt): Bool = groupId < group_num.U

  // Admission and cooperative readiness use B for M/K and A for N. LdA and
  // LdB can simultaneously carry different loops, so treat them as two
  // independent candidates rather than muxing based on whichever is active.
  val lda_candidate = Wire(Vec(nSharers, Bool()))
  val ldb_candidate = Wire(Vec(nSharers, Bool()))
  val lda_ongoing = Wire(Vec(nSharers, Bool()))
  val ldb_ongoing = Wire(Vec(nSharers, Bool()))
  val lda_needs_allocation = Wire(Vec(nSharers, Bool()))
  val ldb_needs_allocation = Wire(Vec(nSharers, Bool()))
  val admission_load = Wire(Vec(nSharers, new CoopLoadState(
    group_w, nSharers, iterator_bitwidth, useVpuFusion)))

  for (i <- 0 until nSharers) {
    val lda = io.in(i).lda
    val ldb = io.in(i).ldb

    lda_candidate(i) := !lda.idle &&
      GemminiISA.PartitionAxis.usesCooperativeLdA(lda.axis)
    ldb_candidate(i) := !ldb.idle &&
      !GemminiISA.PartitionAxis.usesCooperativeLdA(ldb.axis)

    val lda_group = group_data(lda.group_id)
    val ldb_group = group_data(ldb.group_id)
    val lda_group_exists = validGroupId(lda.group_id) && lda_group.valid
    val ldb_group_exists = validGroupId(ldb.group_id) && ldb_group.valid

    lda_ongoing(i) := lda_group_exists &&
      lda_group.bits.axis === lda.axis && lda_group.bits.member_mask(i) &&
      !lda_group.bits.mem_data(i).completed_load_range.valid
    ldb_ongoing(i) := ldb_group_exists &&
      ldb_group.bits.axis === ldb.axis && ldb_group.bits.member_mask(i) &&
      !ldb_group.bits.mem_data(i).completed_load_range.valid

    lda_needs_allocation(i) := lda_candidate(i) && !lda_group_exists
    ldb_needs_allocation(i) := ldb_candidate(i) && !ldb_group_exists

    // One sharer can contribute at most one new group in a cycle. B has fixed
    // priority; after B allocates, A is represented on the following cycle.
    // Existing groups never consume this arbitration point.
    admission_load(i) := Mux(ldb_needs_allocation(i), ldb, lda)

    io.in(i).lda_admission_blocked := lda_candidate(i) && !lda_ongoing(i)
    io.in(i).ldb_admission_blocked := ldb_candidate(i) && !ldb_ongoing(i)

  }

  val lda_idle_delayed = RegNext(
    VecInit(io.in.map(_.lda.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val ldb_idle_delayed = RegNext(
    VecInit(io.in.map(_.ldb.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val ldd_idle_delayed = RegNext(
    VecInit(io.in.map(_.ldd.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val ex_idle_delayed  = RegNext(VecInit(io.in.map(_.ex.idle)),  VecInit(Seq.fill(nSharers)(true.B)))
  val stc_idle_delayed = RegNext(VecInit(io.in.map(_.stc.idle)), VecInit(Seq.fill(nSharers)(true.B)))

  val hasAdmission = VecInit((0 until nSharers).map { i =>
    (ldb_needs_allocation(i) || lda_needs_allocation(i)) &&
      validGroupId(admission_load(i).group_id)
  })
  // The finite group ID is also the physical table index. A request needing
  // allocation therefore always targets its own invalid slot; a free-slot
  // search, age matrix, and winner ranking would only re-encode that ID.
  for (g <- 0 until group_num) {
    val gd = group_data(g)
    val winnerSourceCandidates = VecInit((0 until nSharers).map { sIdx =>
      hasAdmission(sIdx) && admission_load(sIdx).group_id === g.U
    })
    val winnerSourceOH = PriorityEncoderOH(winnerSourceCandidates.asUInt)
    val winnerSource = Mux1H(winnerSourceOH, admission_load)
    val winnerMemberMask = winnerSource.member_mask
    val winnerAxis = winnerSource.axis
    val allocationCandidate = !gd.valid && winnerSourceCandidates.asUInt.orR
    val canAllocate = if (useVpuFusion) {
      val event = io.event_admissions.get(g)
      event.request := allocationCandidate
      event.waitValid := winnerSource.wait_event_valid.get
      event.waitId := winnerSource.wait_event_id.get
      event.produceValid := winnerSource.produce_event_valid.get
      event.produceId := winnerSource.produce_event_id.get
      event.produceSeal := winnerSource.produce_event_seal.get
      event.commit := allocationCandidate && event.ready
      allocationCandidate && event.ready
    } else {
      allocationCandidate
    }

    when (canAllocate) {
      gd.valid := true.B
      gd.bits.axis := winnerAxis
      gd.bits.member_mask := winnerMemberMask

      val group_mask = VecInit(winnerMemberMask.asBools)
      gd.bits.mem_data.zip(group_mask).foreach { case (md, gm) =>
        md.completed_load_range.valid := !gm
        // Only K needs a cross-member initialization barrier. Mark M/N done
        // at allocation to preserve their original completion/lifetime path.
        md.d_load_stream_registered :=
          !gm || winnerAxis =/= GemminiISA.PartitionAxis.K
        md.execute_stream_registered := !gm
        md.store_stream_registered := !gm
        md.completed_load_range.bits.extent := 0.U
        md.completed_load_range.bits.offset := 0.U
      }
      gd.bits.k_overwrite_frontier.valid := false.B
      gd.bits.k_overwrite_frontier.bits.j := 0.U
      gd.bits.k_overwrite_frontier.bits.i := 0.U
      gd.bits.k_store_rr_ptr := 0.U

    }

    if (useVpuFusion) {
      when (canAllocate) {
        gd.bits.produce_event_valid.get :=
          winnerSource.produce_event_valid.get
        gd.bits.produce_event_id.get :=
          winnerSource.produce_event_id.get
      }
    }
  }
  // Update group data when the axis-selected load stream completes. Keeping
  // separate delayed-idle state is essential when LdA and LdB carry different
  // loops concurrently. A selected zero-extent stream is held non-idle by its
  // own blocked input until the group is allocated, then produces this edge.
  for (i <- 0 until nSharers) {
    val lda = io.in(i).lda
    val ldb = io.in(i).ldb
    val ldd = io.in(i).ldd
    val ex = io.in(i).ex
    val stc = io.in(i).stc
    val ldaGroup = group_data(lda.group_id)
    val ldbGroup = group_data(ldb.group_id)
    val lddGroup = group_data(ldd.group_id)
    val exGroup = group_data(ex.group_id)
    val stcGroup = group_data(stc.group_id)

    when (lda.idle && !lda_idle_delayed(i) &&
        GemminiISA.PartitionAxis.usesCooperativeLdA(lda.axis) &&
        validGroupId(lda.group_id) && ldaGroup.valid &&
        ldaGroup.bits.axis === lda.axis && ldaGroup.bits.member_mask(i)) {
      ldaGroup.bits.mem_data(i).completed_load_range.valid := true.B
      ldaGroup.bits.mem_data(i).completed_load_range.bits.extent :=
        lda.owner_extent
      ldaGroup.bits.mem_data(i).completed_load_range.bits.offset :=
        lda.owner_offset
    }

    when (ldb.idle && !ldb_idle_delayed(i) &&
        !GemminiISA.PartitionAxis.usesCooperativeLdA(ldb.axis) &&
        validGroupId(ldb.group_id) && ldbGroup.valid &&
        ldbGroup.bits.axis === ldb.axis && ldbGroup.bits.member_mask(i)) {
      ldbGroup.bits.mem_data(i).completed_load_range.valid := true.B
      ldbGroup.bits.mem_data(i).completed_load_range.bits.extent :=
        ldb.owner_extent
      ldbGroup.bits.mem_data(i).completed_load_range.bits.offset :=
        ldb.owner_offset
    }

    when (ldd.idle && !ldd_idle_delayed(i) &&
        validGroupId(ldd.group_id) && lddGroup.valid &&
        lddGroup.bits.axis === GemminiISA.PartitionAxis.K &&
        lddGroup.bits.axis === ldd.axis && lddGroup.bits.member_mask(i)) {
      lddGroup.bits.mem_data(i).d_load_stream_registered := true.B
    }

    when (ex.idle && !ex_idle_delayed(i) && validGroupId(ex.group_id) &&
        exGroup.valid && exGroup.bits.axis === ex.axis) {
      exGroup.bits.mem_data(i).execute_stream_registered := true.B
    }

    when (stc.idle && !stc_idle_delayed(i) && validGroupId(stc.group_id) &&
        stcGroup.valid && stcGroup.bits.axis === stc.axis) {
      stcGroup.bits.mem_data(i).store_stream_registered := true.B
    }
  }

  // LoopMatmulExecute holds (k,j,i) while it moves from PRELOAD to COMPUTE, so
  // the live COMPUTE phase is the same-tile ownership token. No duplicate pair
  // coordinate needs to be stored in this controller.
  val activeKEx = VecInit(io.in.map(in =>
    !in.ex.idle && in.ex.axis === GemminiISA.PartitionAxis.K))
  val preloadKEx = VecInit(io.in.zip(activeKEx).map { case (in, active) =>
    active && !in.ex.is_compute
  })

  for (i <- 0 until nSharers) {
    val ex = io.in(i).ex
    val gd = group_data(ex.group_id)
    val matched = validGroupId(ex.group_id) && gd.valid &&
      gd.bits.axis === GemminiISA.PartitionAxis.K && gd.bits.member_mask(i)
    val dStreamsRegistered =
      gd.bits.mem_data.map(_.d_load_stream_registered).reduce(_ && _)
    val isFirstOverwritePair = ex.first_k_overwrites &&
      ex.is_first_k_shard && ex.k === 0.U
    val overwriteRegistered = gd.bits.k_overwrite_frontier.valid &&
      tileGe(gd.bits.k_overwrite_frontier.bits.j,
        gd.bits.k_overwrite_frontier.bits.i, ex.j, ex.i)
    val overwriteReady = !ex.first_k_overwrites ||
      isFirstOverwritePair || overwriteRegistered
    val anyComputeAtTile = (0 until nSharers).map { m =>
      val other = io.in(m).ex
      gd.bits.member_mask(m) && activeKEx(m) &&
        other.group_id === ex.group_id && other.is_compute &&
        tileEq(other.j, other.i, ex.j, ex.i)
    }.reduce(_ || _)
    val lowerPreloadContender = if (i == 0) {
      false.B
    } else {
      (0 until i).map { p =>
        val other = io.in(p).ex
        gd.bits.member_mask(p) && preloadKEx(p) &&
          other.group_id === ex.group_id &&
          tileEq(other.j, other.i, ex.j, ex.i)
      }.reduce(_ || _)
    }
    val preloadReady = !anyComputeAtTile &&
      (!lowerPreloadContender || isFirstOverwritePair)
    // A COMPUTE command continues the PRELOAD token already owned by this
    // requester, so only a matching live group is needed in that phase.
    val pairReady = matched && Mux(ex.is_compute, true.B,
      overwriteReady && preloadReady)
    io.in(i).k_execute_ready := !activeKEx(i) ||
      (dStreamsRegistered && pairReady)

    val memberFire = matched && activeKEx(i) && ex.registration_fire
    val isFirstOverwriteCompute = ex.is_compute &&
      ex.first_k_overwrites && ex.is_first_k_shard && ex.k === 0.U
    when (memberFire && isFirstOverwriteCompute) {
      gd.bits.k_overwrite_frontier.valid := true.B
      gd.bits.k_overwrite_frontier.bits.j := ex.j
      gd.bits.k_overwrite_frontier.bits.i := ex.i
    }
  }

  // A K Store-C command may be registered as soon as every member has
  // registered its final-local-K writer for the complete tile range covered
  // by that command. Existing EX->Store address dependencies then wait for
  // actual writer completion. Arbitrate ready member streams round-robin so a
  // continuously ready pod cannot starve behind another pod's Store stream.
  val kStoreGrant = WireInit(VecInit(Seq.fill(nSharers)(false.B)))
  val kStoreReady = Wire(Vec(nSharers, Bool()))
  for (i <- 0 until nSharers) {
    val stc = io.in(i).stc
    val gd = group_data(stc.group_id)
    val lastJWide = stc.global_j +& stc.j_blocks - 1.U
    val lastJ = lastJWide(iterator_bitwidth - 1, 0)
    val allWritersRegistered = gd.bits.mem_data.zipWithIndex.map {
      case (member, m) =>
        val memberEx = io.in(m).ex
        val activeForGroup = activeKEx(m) &&
          memberEx.group_id === stc.group_id
        val completesThisCycle = memberEx.idle && !ex_idle_delayed(m) &&
          memberEx.axis === GemminiISA.PartitionAxis.K &&
          memberEx.group_id === stc.group_id
        // A zero-K member contributes no partial. Wait until its completed
        // descriptor is available before excluding it, so an unregistered
        // member cannot be mistaken for a zero-sized one.
        !gd.bits.member_mask(m) ||
          isEmptyKShard(member) ||
          member.execute_stream_registered || completesThisCycle ||
          (activeForGroup && memberEx.is_last_local_k &&
            tileGt(memberEx.j, memberEx.i, lastJ, stc.global_i))
    }.reduce(_ && _)

    kStoreReady(i) := validGroupId(stc.group_id) && gd.valid &&
      gd.bits.axis === GemminiISA.PartitionAxis.K && gd.bits.member_mask(i) &&
      stc.axis === GemminiISA.PartitionAxis.K && stc.registration_request &&
      stc.j_blocks =/= 0.U && !lastJWide(iterator_bitwidth) &&
      allWritersRegistered
  }

  group_data.zipWithIndex.foreach { case (gd, g) =>
    val requestMask = VecInit((0 until nSharers).map { i =>
      kStoreReady(i) && io.in(i).stc.group_id === g.U
    }).asUInt
    val rotatedRequests = requestMask.rotateRight(gd.bits.k_store_rr_ptr)
    val rotatedWinner = PriorityEncoderOH(rotatedRequests)
    val selectedMask = Mux(requestMask.orR,
      rotatedWinner.rotateLeft(gd.bits.k_store_rr_ptr), 0.U(nSharers.W))

    for (i <- 0 until nSharers) {
      when (selectedMask(i)) {
        kStoreGrant(i) := true.B
      }
    }

    val fireMask = VecInit((0 until nSharers).map { i =>
      selectedMask(i) && io.in(i).stc.registration_fire
    }).asUInt
    // assert(PopCount(fireMask) <= 1.U,
    //   "K Store-C round-robin granted multiple commands")
    when (fireMask.orR) {
      val winner = OHToUInt(fireMask)
      gd.bits.k_store_rr_ptr := Mux(winner === (nSharers - 1).U,
        0.U, winner + 1.U)
    }
  }

  for (i <- 0 until nSharers) {
    when (io.in(i).stc.axis === GemminiISA.PartitionAxis.K) {
      io.in(i).k_store_grant := kStoreGrant(i)
    }
  }

  // A sync group protects only the multi-Gemmini registration window.  Event
  // lifetime is independent: an attached producer completes when this group
  // releases, while the event itself remains ready until its consumer commits.
  group_data.zipWithIndex.foreach { case (gd, g) =>
    val memberStreamsRegistered = allMemberStreamsRegistered(gd.bits.mem_data)
    if (useVpuFusion) {
      val completion = io.event_completions.get(g)
      completion.valid := gd.valid && memberStreamsRegistered &&
        gd.bits.produce_event_valid.get
      completion.id := gd.bits.produce_event_id.get
    }

    when (gd.valid && memberStreamsRegistered) {
      gd.valid := false.B
    }
  }

  // The load iterator is the next command not yet registered in the local RS.
  // A strict lexicographic lead therefore proves that the requested tile has
  // already been allocated. SharedExtEntries then holds Execute until the
  // corresponding LOAD has actually completed.
  def inHalfOpenRange(coord: UInt, offset: UInt, extent: UInt): Bool = {
    val wideCoord = coord.pad(iterator_bitwidth + 1)
    val wideOffset = offset.pad(iterator_bitwidth + 1)
    val wideEnd = offset +& extent
    wideCoord >= wideOffset && wideCoord < wideEnd
  }

  def lexAhead(loadOuter: UInt, loadInner: UInt,
               exOuter: UInt, exInner: UInt): Bool =
    loadOuter > exOuter || (loadOuter === exOuter && loadInner > exInner)

  // Common M/N cooperative-load readiness. M owns B/K while N owns A/I. K
  // retains its B/K admission descriptor, although the K scheduler ignores
  // this signal in favor of its group grant.
  for (i <- 0 until nSharers) {
    val ex = io.in(i).ex
    val groupMaskForEx = VecInit(ex.member_mask.asBools)
    val exGlobalK = ex.k.pad(iterator_bitwidth + 1)
    val useLdA = GemminiISA.PartitionAxis.usesCooperativeLdA(ex.axis)
    val ownerCoord = Mux(useLdA,
      ex.i.pad(iterator_bitwidth + 1),
      exGlobalK)

    val exGroup = group_data(ex.group_id)
    val completedOwnerHasOperand = validGroupId(ex.group_id) &&
      exGroup.valid && exGroup.bits.axis === ex.axis &&
      exGroup.bits.mem_data.zip(groupMaskForEx).map {
        case (member, selected) =>
          selected && member.completed_load_range.valid &&
            inHalfOpenRange(ownerCoord,
              member.completed_load_range.bits.offset,
              member.completed_load_range.bits.extent)
      }.reduce(_ || _)

    val liveOwnerHasOperand = io.in.zip(groupMaskForEx).map {
      case (candidate, selected) =>
      val load = Mux(useLdA, candidate.lda, candidate.ldb)
      val loadGlobalOuter = Mux(useLdA,
        load.next_outer.pad(iterator_bitwidth + 1),
        load.owner_offset +& load.next_outer)
      val loadGlobalInner = Mux(useLdA,
        load.owner_offset +& load.next_inner,
        load.next_inner.pad(iterator_bitwidth + 1))
      val exInner = Mux(useLdA,
        ex.i.pad(iterator_bitwidth + 1), ex.j.pad(iterator_bitwidth + 1))

      selected && !load.idle && load.axis === ex.axis &&
        load.group_id === ex.group_id &&
        inHalfOpenRange(ownerCoord, load.owner_offset, load.owner_extent) &&
        lexAhead(loadGlobalOuter, loadGlobalInner, exGlobalK, exInner)
    }.reduce(_ || _)

    io.in(i).coop_operand_registered :=
      completedOwnerHasOperand || liveOwnerHasOperand
  }

  // Invalid entries are never observed; allocation initializes every field
  // before setting valid, so reset only needs to invalidate the slots.
  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
  }
}
