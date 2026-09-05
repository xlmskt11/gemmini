package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

class LoopMatmulGroupControllerTester(c: LoopMatmulGroupController)
    extends PeekPokeTester(c) {
  private val nSharers = 4

  private def driveLoad(load: CoopLoadState, groupId: Int = 0,
                        memberMask: Int = 0, idle: Boolean = true): Unit = {
    poke(load.axis, GemminiISA.PartitionAxis.M.litValue)
    poke(load.group_id, groupId)
    poke(load.member_mask, memberMask)
    poke(load.wait_event_valid.get, false)
    poke(load.wait_event_id.get, 0)
    poke(load.produce_event_valid.get, memberMask != 0)
    poke(load.produce_event_id.get, 5)
    poke(load.produce_event_seal.get, memberMask != 0)
    poke(load.next_outer, 0)
    poke(load.next_inner, 0)
    poke(load.owner_offset, 0)
    poke(load.owner_extent, if (memberMask == 0) 0 else 1)
    poke(load.idle, idle)
  }

  private def driveMember(i: Int, groupId: Int = 0, memberMask: Int = 0,
                          idle: Boolean = true): Unit = {
    driveLoad(c.io.in(i).lda)
    driveLoad(c.io.in(i).ldb, groupId, memberMask, idle)

    poke(c.io.in(i).ldd.axis, GemminiISA.PartitionAxis.M.litValue)
    poke(c.io.in(i).ldd.group_id, groupId)
    poke(c.io.in(i).ldd.idle, true)

    poke(c.io.in(i).ex.axis, GemminiISA.PartitionAxis.M.litValue)
    poke(c.io.in(i).ex.group_id, groupId)
    poke(c.io.in(i).ex.member_mask, memberMask)
    poke(c.io.in(i).ex.k, 0)
    poke(c.io.in(i).ex.j, 0)
    poke(c.io.in(i).ex.i, 0)
    poke(c.io.in(i).ex.is_first_k_shard, false)
    poke(c.io.in(i).ex.is_compute, false)
    poke(c.io.in(i).ex.registration_fire, false)
    poke(c.io.in(i).ex.is_last_local_k, false)
    poke(c.io.in(i).ex.first_k_overwrites, false)
    poke(c.io.in(i).ex.idle, idle)

    poke(c.io.in(i).stc.axis, GemminiISA.PartitionAxis.M.litValue)
    poke(c.io.in(i).stc.group_id, groupId)
    poke(c.io.in(i).stc.global_j, 0)
    poke(c.io.in(i).stc.global_i, 0)
    poke(c.io.in(i).stc.j_blocks, 1)
    poke(c.io.in(i).stc.registration_request, false)
    poke(c.io.in(i).stc.registration_fire, false)
    poke(c.io.in(i).stc.idle, idle)
  }

  for (g <- 0 until nSharers * 2) {
    poke(c.io.event_admissions.get(g).ready, true)
  }
  for (i <- 0 until nSharers) driveMember(i)

  // Sync-group allocation is the event admission point. One producer is
  // attached for the whole group, independent of its Gemmini member count.
  driveMember(0, groupId = 0, memberMask = 0x3, idle = false)
  driveMember(1, groupId = 0, memberMask = 0x3, idle = false)
  expect(c.io.event_admissions.get(0).request, true)
  expect(c.io.event_admissions.get(0).commit, true)
  expect(c.io.event_admissions.get(0).produceValid, true)
  expect(c.io.event_admissions.get(0).produceId, 5)
  expect(c.io.event_admissions.get(0).produceSeal, true)
  step(1)

  // The producer completes when all child streams have registered. VPU
  // commands no longer participate in the sync-group lifetime.
  driveMember(0, groupId = 0, memberMask = 0x3, idle = true)
  driveMember(1, groupId = 0, memberMask = 0x3, idle = true)
  step(1)
  expect(c.io.event_completions.get(0).valid, true)
  expect(c.io.event_completions.get(0).id, 5)
  step(1)
  expect(c.io.event_completions.get(0).valid, false)

  // The finite sync ID is immediately reusable after registration completes.
  driveMember(0, groupId = 0, memberMask = 0x1, idle = false)
  for (i <- 1 until nSharers) driveMember(i)
  expect(c.io.event_admissions.get(0).request, true)
  expect(c.io.event_admissions.get(0).commit, true)
}

class LoopMatmulGroupControllerUnitTest extends ChiselFlatSpec {
  behavior of "LoopMatmulGroupController"

  it should "release sync groups at registration and signal EventTracker" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/ldb-complete-control"
    )

    chisel3.iotesters.Driver.execute(args, () => new LoopMatmulGroupController(
      nSharers = 4, useVpuFusion = true)) {
      c => new LoopMatmulGroupControllerTester(c)
    } should be (true)
  }
}
