package gemmini

import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

abstract class ExtLoopMatmulControlTester(c: LoopMatmulGroupController)
    extends PeekPokeTester(c) {
  protected val nSharers = 4

  protected def idleLoad(load: CoopLoadState, axis: Int = 0): Unit = {
    poke(load.axis, axis)
    poke(load.group_id, 0)
    poke(load.member_mask, 0)
    poke(load.next_outer, 0)
    poke(load.next_inner, 0)
    poke(load.owner_offset, 0)
    poke(load.owner_extent, 0)
    poke(load.idle, true)
  }

  protected def idleMember(i: Int): Unit = {
    idleLoad(c.io.in(i).lda)
    idleLoad(c.io.in(i).ldb)

    poke(c.io.in(i).ldd.axis, 0)
    poke(c.io.in(i).ldd.group_id, 0)
    poke(c.io.in(i).ldd.idle, true)

    poke(c.io.in(i).ex.axis, 0)
    poke(c.io.in(i).ex.group_id, 0)
    poke(c.io.in(i).ex.member_mask, 0)
    poke(c.io.in(i).ex.k, 0)
    poke(c.io.in(i).ex.j, 0)
    poke(c.io.in(i).ex.i, 0)
    poke(c.io.in(i).ex.is_first_k_shard, false)
    poke(c.io.in(i).ex.is_compute, false)
    poke(c.io.in(i).ex.registration_fire, false)
    poke(c.io.in(i).ex.is_last_local_k, false)
    poke(c.io.in(i).ex.first_k_overwrites, false)
    poke(c.io.in(i).ex.idle, true)

    poke(c.io.in(i).stc.axis, 0)
    poke(c.io.in(i).stc.group_id, 0)
    poke(c.io.in(i).stc.global_j, 0)
    poke(c.io.in(i).stc.global_i, 0)
    poke(c.io.in(i).stc.j_blocks, 1)
    poke(c.io.in(i).stc.registration_request, false)
    poke(c.io.in(i).stc.registration_fire, false)
    poke(c.io.in(i).stc.idle, true)
  }

  protected def idleAll(): Unit =
    for (i <- 0 until nSharers) idleMember(i)

  protected def activeLoad(load: CoopLoadState, axis: Int, groupId: Int,
                           mask: Int, outer: Int, inner: Int,
                           auxOffset: Int, auxExtent: Int): Unit = {
    poke(load.axis, axis)
    poke(load.group_id, groupId)
    poke(load.member_mask, mask)
    poke(load.next_outer, outer)
    poke(load.next_inner, inner)
    poke(load.owner_offset, auxOffset)
    poke(load.owner_extent, auxExtent)
    poke(load.idle, false)
  }

  protected def activeEx(i: Int, axis: Int, groupId: Int, mask: Int,
                         j: Int, row: Int, phase: Int,
                         finalLocalK: Boolean,
                         k: Int = 0, kOffset: Int = 0,
                         firstOverwriteMode: Boolean = false): Unit = {
    poke(c.io.in(i).ex.axis, axis)
    poke(c.io.in(i).ex.group_id, groupId)
    poke(c.io.in(i).ex.member_mask, mask)
    poke(c.io.in(i).ex.k, k)
    poke(c.io.in(i).ex.j, j)
    poke(c.io.in(i).ex.i, row)
    poke(c.io.in(i).ex.is_first_k_shard, kOffset == 0)
    poke(c.io.in(i).ex.is_compute, phase != 0)
    poke(c.io.in(i).ex.registration_fire, false)
    poke(c.io.in(i).ex.is_last_local_k, finalLocalK)
    poke(c.io.in(i).ex.first_k_overwrites, firstOverwriteMode)
    poke(c.io.in(i).ex.idle, false)
  }

  protected def activeStc(i: Int, axis: Int, groupId: Int,
                          j: Int, row: Int, blocks: Int): Unit = {
    poke(c.io.in(i).stc.axis, axis)
    poke(c.io.in(i).stc.group_id, groupId)
    poke(c.io.in(i).stc.global_j, j)
    poke(c.io.in(i).stc.global_i, row)
    poke(c.io.in(i).stc.j_blocks, blocks)
    poke(c.io.in(i).stc.registration_request, true)
    poke(c.io.in(i).stc.registration_fire, false)
    poke(c.io.in(i).stc.idle, false)
  }

  protected def pulseStcFire(i: Int): Unit = {
    poke(c.io.in(i).stc.registration_fire, true)
    step(1)
    poke(c.io.in(i).stc.registration_fire, false)
  }

  protected def pulseExFire(i: Int): Unit = {
    poke(c.io.in(i).ex.registration_fire, true)
    step(1)
    poke(c.io.in(i).ex.registration_fire, false)
  }

  protected def registerKDataLoadStreams(groupId: Int, mask: Int): Unit = {
    for (i <- 0 until nSharers if ((mask >> i) & 1) != 0) {
      poke(c.io.in(i).ldd.axis, GemminiISA.PartitionAxis.K.litValue)
      poke(c.io.in(i).ldd.group_id, groupId)
      poke(c.io.in(i).ldd.idle, false)
    }
    step(1)
    for (i <- 0 until nSharers if ((mask >> i) & 1) != 0) {
      poke(c.io.in(i).ldd.idle, true)
    }
    step(1)
  }
}

class ExtNStreamingTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisN = 1
  private val groupId = 0
  private val mask = 0x5 // physical pods 0 and 2

  idleAll()

  // Allocate an N group. Pod 0 owns I=[0,2), while pod 2 owns I=[2,4).
  activeLoad(c.io.in(0).lda, axisN, groupId, mask,
    outer = 0, inner = 2, auxOffset = 0, auxExtent = 2)
  activeLoad(c.io.in(2).lda, axisN, groupId, mask,
    outer = 0, inner = 0, auxOffset = 2, auxExtent = 2)
  step(1)

  // The owner of I=1 is lexically ahead: (k=0, global-i=2) > (0,1).
  activeEx(0, axisN, groupId, mask, j = 0, row = 1,
    phase = 0, finalLocalK = false)
  expect(c.io.in(0).coop_operand_registered, true)

  // I=2 belongs to pod 2. Its cursor is exactly (0,2), not strictly ahead;
  // pod 0's later cursor cannot release a row outside pod 0's owner range.
  poke(c.io.in(0).ex.i, 2)
  expect(c.io.in(0).coop_operand_registered, false)

  // Advancing only the owner cursor releases I=2 without waiting for all A.
  poke(c.io.in(2).lda.next_inner, 1)
  expect(c.io.in(0).coop_operand_registered, true)

  // Once pod 2 closes its stream, the recorded [2,4) completion range takes
  // over. Pod 0 deliberately remains live, proving this is not an all-A gate.
  poke(c.io.in(2).lda.idle, true)
  step(1)
  poke(c.io.in(0).ex.i, 3)
  expect(c.io.in(0).coop_operand_registered, true)
  expect(c.io.in(0).lda.idle, false)
}

class ExtMStreamingTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisM = 0
  private val groupId = 0
  private val mask = 0x5 // physical pods 0 and 2

  idleAll()

  // M keeps compute partitioning on I, while B load ownership is split over
  // K. Pod 0 owns K=[0,2), and pod 2 owns K=[2,4).
  activeLoad(c.io.in(0).ldb, axisM, groupId, mask,
    outer = 2, inner = 0, auxOffset = 0, auxExtent = 2)
  activeLoad(c.io.in(2).ldb, axisM, groupId, mask,
    outer = 0, inner = 0, auxOffset = 2, auxExtent = 2)
  step(1)

  activeEx(0, axisM, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = false)
  poke(c.io.in(0).ex.k, 1)
  expect(c.io.in(0).coop_operand_registered, true)

  // K=2 belongs to pod 2. Its cursor is exactly (global-k=2, j=0), so the
  // strict ahead relation is false; pod 0 cannot release another K owner.
  poke(c.io.in(0).ex.k, 2)
  expect(c.io.in(0).coop_operand_registered, false)

  // Advancing only pod 2's B/K cursor releases this tile without requiring
  // the complete shared-B stream to finish.
  poke(c.io.in(2).ldb.next_inner, 1)
  expect(c.io.in(0).coop_operand_registered, true)

  // A completed K owner range remains sufficient while another owner is live.
  poke(c.io.in(2).ldb.idle, true)
  step(1)
  poke(c.io.in(0).ex.k, 3)
  expect(c.io.in(0).coop_operand_registered, true)
  expect(c.io.in(0).ldb.idle, false)
}

class ExtKPairAndTailTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2
  private val groupId = 1
  private val mask = 0xa // non-contiguous physical pods 1 and 3

  idleAll()

  activeLoad(c.io.in(1).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(3).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 1, auxExtent = 1)
  step(1)
  registerKDataLoadStreams(groupId, mask)

  // Simultaneous PRELOAD contenders for C[0,0]: lowest physical ID wins.
  activeEx(1, axisK, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = true)
  activeEx(3, axisK, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = true)
  expect(c.io.in(1).k_execute_ready, true)
  expect(c.io.in(3).k_execute_ready, false)

  // Pod 1's matching COMPUTE phase keeps same-tile priority; pod 3 cannot
  // interpose between the two phases.
  pulseExFire(1)
  poke(c.io.in(1).ex.is_compute, 1)
  expect(c.io.in(1).k_execute_ready, true)
  expect(c.io.in(3).k_execute_ready, false)
  pulseExFire(1)

  // Pod 1 advances to a different C tile. Pod 3 can now write C[0,0], because
  // pod 1's Execute cursor is beyond that tile.
  poke(c.io.in(1).ex.is_compute, 0)
  poke(c.io.in(1).ex.i, 1)
  expect(c.io.in(1).k_execute_ready, true)
  expect(c.io.in(3).k_execute_ready, true)

  pulseExFire(3)
  poke(c.io.in(3).ex.is_compute, 1)
  expect(c.io.in(3).k_execute_ready, true)
  // A different tile remains independently eligible while pod 3 finishes
  // C[0,0], demonstrating that arbitration is not a group-wide barrier.
  expect(c.io.in(1).k_execute_ready, true)
  pulseExFire(3)

  // Both members have now moved beyond the sealed tile. At C[0,1] the same
  // non-contiguous-mask priority rule applies.
  poke(c.io.in(3).ex.is_compute, 0)
  poke(c.io.in(3).ex.i, 1)
  expect(c.io.in(1).k_execute_ready, true)
  expect(c.io.in(3).k_execute_ready, false)
}

class ExtKReleaseTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2
  private val groupId = 2
  private val mask = 0x1

  idleAll()

  activeLoad(c.io.in(0).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, false)

  // Close the selected load stream.
  poke(c.io.in(0).ldb.idle, true)
  step(1)

  // Register the K D-load stream before Execute admission.
  registerKDataLoadStreams(groupId, mask)

  // Register one final-K pair; the Execute stream must close cleanly.
  activeEx(0, axisK, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = true)
  expect(c.io.in(0).k_execute_ready, true)
  pulseExFire(0)
  poke(c.io.in(0).ex.is_compute, 1)
  pulseExFire(0)
  poke(c.io.in(0).ex.idle, true)
  step(1)

  // Register Store-C and let the group retire.
  poke(c.io.in(0).stc.axis, axisK)
  poke(c.io.in(0).stc.group_id, groupId)
  poke(c.io.in(0).stc.idle, false)
  step(1)
  poke(c.io.in(0).stc.idle, true)
  step(1)

  // Reusing the same finite group ID must allocate fresh state; stale K
  // progress or false-intent state would prevent this transition.
  activeLoad(c.io.in(0).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  expect(c.io.in(0).ldb_admission_blocked, true)
  // The first edge retires the old valid slot; the following edge allocates
  // the still-held admission candidate under the reused ID.
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, true)
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, false)
}

class ExtKMultipleLiveAdmissionTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisM = 0
  private val axisN = 1
  private val axisK = 2

  idleAll()

  // Allocate one K group and leave it live after its anchor load closes.
  activeLoad(c.io.in(0).ldb, axisK, groupId = 0, mask = 0x1,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, false)

  // Close only its anchor stream; LdD/EX/StC remain open at group level.
  poke(c.io.in(0).ldb.idle, true)
  step(1)

  // A second, disjoint K group and independent M/N groups may all allocate
  // while K group 0 remains live.
  activeLoad(c.io.in(1).ldb, axisK, groupId = 1, mask = 0x2,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(2).ldb, axisM, groupId = 2, mask = 0x4,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(3).lda, axisN, groupId = 3, mask = 0x8,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  expect(c.io.in(1).ldb_admission_blocked, true)
  expect(c.io.in(2).ldb_admission_blocked, true)
  expect(c.io.in(3).lda_admission_blocked, true)
  step(1)
  expect(c.io.in(1).ldb_admission_blocked, false)
  expect(c.io.in(2).ldb_admission_blocked, false)
  expect(c.io.in(3).lda_admission_blocked, false)

  // Both K groups can independently reach Execute, even at the same logical
  // tile, because different group IDs refer to disjoint accumulator ranges.
  registerKDataLoadStreams(groupId = 0, mask = 0x1)
  registerKDataLoadStreams(groupId = 1, mask = 0x2)

  activeEx(0, axisK, groupId = 0, mask = 0x1, j = 0, row = 0,
    phase = 0, finalLocalK = false)
  activeEx(1, axisK, groupId = 1, mask = 0x2, j = 0, row = 0,
    phase = 0, finalLocalK = false)
  expect(c.io.in(0).k_execute_ready, true)
  expect(c.io.in(1).k_execute_ready, true)

  // Complete only group 0's Execute stream. Group 1 deliberately remains
  // live and must not affect group 0's lifecycle.
  pulseExFire(0)
  poke(c.io.in(0).ex.is_compute, true)
  pulseExFire(0)
  poke(c.io.in(0).ex.idle, true)
  step(1)

  poke(c.io.in(0).stc.axis, axisK)
  poke(c.io.in(0).stc.group_id, 0)
  poke(c.io.in(0).stc.idle, false)
  step(1)
  poke(c.io.in(0).stc.idle, true)
  step(1)

  // Reuse group ID 0 while group 1 remains live. One edge retires the old
  // slot and the next admits the still-held replacement descriptor.
  activeLoad(c.io.in(0).ldb, axisK, groupId = 0, mask = 0x1,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  expect(c.io.in(0).ldb_admission_blocked, true)
  expect(c.io.in(1).ldb_admission_blocked, false)
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, true)
  expect(c.io.in(1).ldb_admission_blocked, false)
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, false)
  expect(c.io.in(1).ldb_admission_blocked, false)
  expect(c.io.in(2).ldb_admission_blocked, false)
  expect(c.io.in(3).lda_admission_blocked, false)
}

class ExtKSameCycleAdmissionTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2

  idleAll()

  // Two unrelated K IDs arrive together. Both producers initially observe
  // backpressure until the allocation edge.
  activeLoad(c.io.in(0).ldb, axisK, groupId = 0, mask = 0x1,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(1).ldb, axisK, groupId = 1, mask = 0x2,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  expect(c.io.in(0).ldb_admission_blocked, true)
  expect(c.io.in(1).ldb_admission_blocked, true)
  step(1)

  expect(c.io.in(0).ldb_admission_blocked, false)
  expect(c.io.in(1).ldb_admission_blocked, false)

  // Both descriptors stay admitted on the following cycle, demonstrating
  // that neither K ID suppresses the other after simultaneous allocation.
  step(1)
  expect(c.io.in(0).ldb_admission_blocked, false)
  expect(c.io.in(1).ldb_admission_blocked, false)
}

class ExtKFirstOverwriteTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2
  private val groupId = 4
  private val mask = 0xa // higher physical-ID pod 3 is the overwrite owner

  idleAll()

  activeLoad(c.io.in(1).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 1, auxExtent = 1)
  activeLoad(c.io.in(3).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  step(1)
  registerKDataLoadStreams(groupId, mask)

  // A non-owner cannot register a partial before the offset-zero member has
  // registered the matching local-k-zero overwrite pair.
  activeEx(1, axisK, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = false,
    k = 0, kOffset = 1, firstOverwriteMode = true)
  activeEx(3, axisK, groupId, mask, j = 0, row = 0,
    phase = 0, finalLocalK = false,
    k = 0, kOffset = 0, firstOverwriteMode = true)
  expect(c.io.in(1).k_execute_ready, false)
  // The overwrite owner bypasses the lower physical-ID PRELOAD contender.
  expect(c.io.in(3).k_execute_ready, true)

  pulseExFire(3)
  expect(c.io.in(1).k_execute_ready, false)
  poke(c.io.in(3).ex.is_compute, 1)
  pulseExFire(3)

  // The overwrite COMPUTE registration advances the tile watermark. Actual
  // execution remains ordered by the shared ACC dependency table.
  poke(c.io.in(3).ex.is_compute, 0)
  poke(c.io.in(3).ex.i, 1)
  expect(c.io.in(1).k_execute_ready, true)
}

class ExtKStoreProgressRrTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2
  private val groupId = 5
  private val mask = 0x3

  idleAll()

  activeLoad(c.io.in(0).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(1).ldb, axisK, groupId, mask,
    outer = 0, inner = 0, auxOffset = 1, auxExtent = 1)
  step(1)
  registerKDataLoadStreams(groupId, mask)

  def registerFinalPair(pod: Int, j: Int): Unit = {
    activeEx(pod, axisK, groupId, mask, j = j, row = 0,
      phase = 0, finalLocalK = true,
      k = 0, kOffset = pod, firstOverwriteMode = false)
    expect(c.io.in(pod).k_execute_ready, true)
    pulseExFire(pod)
    poke(c.io.in(pod).ex.is_compute, 1)
    pulseExFire(pod)
  }

  // Both members have registered through J=2 and point at an unregistered
  // J=3 pair. A blocks=4 Store beginning at J=0 therefore cannot proceed.
  registerFinalPair(0, j = 2)
  poke(c.io.in(0).ex.is_compute, 0)
  poke(c.io.in(0).ex.j, 3)
  registerFinalPair(1, j = 2)
  poke(c.io.in(1).ex.is_compute, 0)
  poke(c.io.in(1).ex.j, 3)

  activeStc(0, axisK, groupId, j = 0, row = 0, blocks = 4)
  activeStc(1, axisK, groupId, j = 0, row = 0, blocks = 4)
  expect(c.io.in(0).k_store_grant, false)
  expect(c.io.in(1).k_store_grant, false)

  // Register both members' final pair at the command's last tile.
  expect(c.io.in(0).k_execute_ready, true)
  pulseExFire(0)
  poke(c.io.in(0).ex.is_compute, 1)
  pulseExFire(0)
  // This was the member's final tile. LoopMatmulExecute leaves COMPUTE after
  // the fire; do not leave a synthetic COMPUTE request active without its
  // just-consumed PRELOAD token while the other member advances.
  poke(c.io.in(0).ex.idle, true)
  expect(c.io.in(1).k_execute_ready, true)
  pulseExFire(1)
  poke(c.io.in(1).ex.is_compute, 1)
  pulseExFire(1)
  poke(c.io.in(1).ex.idle, true)

  // Pointer starts at pod 0 and advances only when the granted command fires.
  expect(c.io.in(0).k_store_grant, true)
  expect(c.io.in(1).k_store_grant, false)
  pulseStcFire(0)
  expect(c.io.in(0).k_store_grant, false)
  expect(c.io.in(1).k_store_grant, true)

  // A range which crosses the 16-bit J boundary must fail closed instead of
  // wrapping its endpoint to a small coordinate and granting too early.
  poke(c.io.in(0).stc.global_j, 65535)
  poke(c.io.in(0).stc.j_blocks, 2)
  poke(c.io.in(1).stc.global_j, 65535)
  poke(c.io.in(1).stc.j_blocks, 2)
  expect(c.io.in(0).k_store_grant, false)
  expect(c.io.in(1).k_store_grant, false)
}

class ExtKIndependentStoreGroupsTester(c: LoopMatmulGroupController)
    extends ExtLoopMatmulControlTester(c) {
  private val axisK = 2

  idleAll()

  // The two two-member K groups use disjoint accumulator ranges even though
  // their local tile coordinates are equal.
  activeLoad(c.io.in(0).ldb, axisK, groupId = 0, mask = 0x3,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(1).ldb, axisK, groupId = 0, mask = 0x3,
    outer = 0, inner = 0, auxOffset = 1, auxExtent = 1)
  activeLoad(c.io.in(2).ldb, axisK, groupId = 1, mask = 0xc,
    outer = 0, inner = 0, auxOffset = 0, auxExtent = 1)
  activeLoad(c.io.in(3).ldb, axisK, groupId = 1, mask = 0xc,
    outer = 0, inner = 0, auxOffset = 1, auxExtent = 1)
  step(1)
  for (i <- 0 until nSharers) poke(c.io.in(i).ldb.idle, true)
  step(1)

  registerKDataLoadStreams(groupId = 0, mask = 0x3)
  registerKDataLoadStreams(groupId = 1, mask = 0xc)

  def registerFinalPairAndClose(pod: Int, groupId: Int, mask: Int): Unit = {
    activeEx(pod, axisK, groupId, mask, j = 0, row = 0,
      phase = 0, finalLocalK = true)
    expect(c.io.in(pod).k_execute_ready, true)
    pulseExFire(pod)
    poke(c.io.in(pod).ex.is_compute, true)
    expect(c.io.in(pod).k_execute_ready, true)
    pulseExFire(pod)
    poke(c.io.in(pod).ex.idle, true)
    step(1)
  }

  registerFinalPairAndClose(pod = 0, groupId = 0, mask = 0x3)
  registerFinalPairAndClose(pod = 1, groupId = 0, mask = 0x3)
  registerFinalPairAndClose(pod = 2, groupId = 1, mask = 0xc)
  registerFinalPairAndClose(pod = 3, groupId = 1, mask = 0xc)

  // Each group performs its own progress check and RR selection, so both
  // groups select their first Store stream in the same cycle.
  activeStc(0, axisK, groupId = 0, j = 0, row = 0, blocks = 1)
  activeStc(1, axisK, groupId = 0, j = 0, row = 0, blocks = 1)
  activeStc(2, axisK, groupId = 1, j = 0, row = 0, blocks = 1)
  activeStc(3, axisK, groupId = 1, j = 0, row = 0, blocks = 1)
  expect(c.io.in(0).k_store_grant, true)
  expect(c.io.in(1).k_store_grant, false)
  expect(c.io.in(2).k_store_grant, true)
  expect(c.io.in(3).k_store_grant, false)

  poke(c.io.in(0).stc.registration_fire, true)
  poke(c.io.in(2).stc.registration_fire, true)
  step(1)
  poke(c.io.in(0).stc.registration_fire, false)
  poke(c.io.in(2).stc.registration_fire, false)

  // The two independent RR pointers advance from pod 0 to 1 and from pod 2
  // to 3; neither group changes the other group's winner.
  expect(c.io.in(0).k_store_grant, false)
  expect(c.io.in(1).k_store_grant, true)
  expect(c.io.in(2).k_store_grant, false)
  expect(c.io.in(3).k_store_grant, true)
}

class ExtLoopMatmulControlUnitTest extends ChiselFlatSpec {
  behavior of "LoopMatmulGroupController axis and K ordering"

  it should "release only the M B/K owner cursor and use completion ranges" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-m-streaming")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtMStreamingTester(c)
    } should be (true)
  }

  it should "release only the N owner cursor and use completion ranges" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-n-streaming")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtNStreamingTester(c)
    } should be (true)
  }

  it should "serialize same-tile K pairs with skewed member progress" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-pair-tail")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKPairAndTailTester(c)
    } should be (true)
  }

  it should "release and reuse a completed K group" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-no-cto-release")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKReleaseTester(c)
    } should be (true)
  }

  it should "keep multiple K groups live without blocking M or N" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-multiple-live")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKMultipleLiveAdmissionTester(c)
    } should be (true)
  }

  it should "admit two disjoint K group IDs in the same cycle" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-same-cycle")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKSameCycleAdmissionTester(c)
    } should be (true)
  }

  it should "release K partials only after the first overwrite pair registers" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-first-overwrite")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKFirstOverwriteTester(c)
    } should be (true)
  }

  it should "gate the full Store range and grant ready K stores round-robin" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-store-progress-rr")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKStoreProgressRrTester(c)
    } should be (true)
  }

  it should "grant Store independently for disjoint live K groups" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/ext-loop-k-independent-store-groups")
    chisel3.iotesters.Driver.execute(args,
      () => new LoopMatmulGroupController(nSharers = 4, useVpuFusion = false)) {
      c => new ExtKIndependentStoreGroupsTester(c)
    } should be (true)
  }
}
