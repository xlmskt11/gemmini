package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

class FusionLoopMatmulGroupControllerTester(
  c: FusionLoopMatmulGroupController
) extends PeekPokeTester(c) {
  private val nSharers = 2

  private def driveMember(
    member: Int,
    groupId: Int = 0,
    memberMask: Int = 0,
    hasFollowup: Boolean = false,
    idle: Boolean = true,
    completionValid: Boolean = false
  ): Unit = {
    poke(c.io.in(member).ldb.group_id, groupId)
    poke(c.io.in(member).ldb.member_mask, memberMask)
    poke(c.io.in(member).ldb.has_gemv_followup, hasFollowup)
    poke(c.io.in(member).ldb.idle, idle)
    poke(c.io.in(member).completion.valid, completionValid)
    poke(c.io.in(member).completion.bits, groupId)
  }

  private def driveGemv(
    groupId: Int = 0,
    pending: Boolean = false,
    lastDispatchFire: Boolean = false,
    abort: Boolean = false
  ): Unit = {
    poke(c.io.gemv.group_id, groupId)
    poke(c.io.gemv.pending, pending)
    poke(c.io.gemv.last_dispatch_fire, lastDispatchFire)
    poke(c.io.gemv.abort, abort)
  }

  for (member <- 0 until nSharers) driveMember(member)
  driveGemv()

  // The first active LdB anchors and allocates group 3. Admission remains
  // blocked in that presentation cycle and opens from the allocated table on
  // the following cycle.
  driveMember(0, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = false)
  driveMember(1, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = false)
  driveGemv(groupId = 3, pending = true)
  expect(c.io.in(0).ldb_admission_blocked, true)
  expect(c.io.in(1).ldb_admission_blocked, true)
  step(1)

  expect(c.io.in(0).ldb_admission_blocked, false)
  expect(c.io.in(1).ldb_admission_blocked, false)
  expect(c.io.gemv.group_allocated, true)
  expect(c.io.gemv.dispatch_enable, false)
  expect(c.io.gemv.dispatch_reject, false)

  // Closing LdB is only the admission-anchor completion. It must not enable
  // VPU dispatch before both LoopMatmul instances report that all child
  // command streams have entered their local reservation stations.
  driveMember(0, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  driveMember(1, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  step(1)
  expect(c.io.gemv.dispatch_enable, false)

  driveMember(0, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true, completionValid = true)
  driveMember(1, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  step(1)
  expect(c.io.gemv.dispatch_enable, false)

  driveMember(0, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  driveMember(1, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true, completionValid = true)
  step(1)
  expect(c.io.gemv.dispatch_enable, true)

  // The table entry is still live until the final grouped VPU command fires;
  // a new LdB which tries to reuse group 3 cannot be admitted meanwhile.
  driveMember(0, groupId = 3, memberMask = 0x1,
    hasFollowup = false, idle = false)
  driveMember(1)
  expect(c.io.in(0).ldb_admission_blocked, true)

  driveMember(0, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  driveMember(1, groupId = 3, memberMask = 0x3,
    hasFollowup = true, idle = true)
  driveGemv(groupId = 3, pending = true, lastDispatchFire = true)
  step(1)
  driveGemv(groupId = 3, pending = true)
  expect(c.io.gemv.group_allocated, false)
  expect(c.io.gemv.dispatch_enable, false)

  // A group without a VPU follow-up releases automatically after its member
  // completion; no VPU dispatch handshake is required.
  driveMember(0, groupId = 5, memberMask = 0x1,
    hasFollowup = false, idle = false)
  driveMember(1)
  driveGemv(groupId = 5, pending = true)
  step(1)
  expect(c.io.gemv.group_allocated, true)
  expect(c.io.gemv.dispatch_enable, false)
  expect(c.io.gemv.dispatch_reject, true)

  driveMember(0, groupId = 5, memberMask = 0x1,
    hasFollowup = false, idle = true, completionValid = true)
  step(1)
  expect(c.io.gemv.group_allocated, true)
  step(1)
  expect(c.io.gemv.group_allocated, false)
}

class FusionLoopMatmulSingleMemberTester(
  c: FusionLoopMatmulGroupController
) extends PeekPokeTester(c) {
  poke(c.io.in(0).ldb.group_id, 7)
  poke(c.io.in(0).ldb.member_mask, 1)
  poke(c.io.in(0).ldb.has_gemv_followup, true)
  poke(c.io.in(0).ldb.idle, false)
  poke(c.io.in(0).completion.valid, false)
  poke(c.io.in(0).completion.bits, 7)
  poke(c.io.gemv.pending, true)
  poke(c.io.gemv.group_id, 7)
  poke(c.io.gemv.last_dispatch_fire, false)
  poke(c.io.gemv.abort, false)

  step(1)
  expect(c.io.gemv.group_allocated, true)
  expect(c.io.in(0).ldb_admission_blocked, false)

  poke(c.io.in(0).ldb.idle, true)
  step(1)
  poke(c.io.in(0).completion.valid, true)
  step(1)
  expect(c.io.gemv.dispatch_enable, true)

  poke(c.io.in(0).completion.valid, false)
  poke(c.io.gemv.last_dispatch_fire, true)
  step(1)
  expect(c.io.gemv.group_allocated, false)
}

class FusionLoopMatmulGroupControllerUnitTest extends ChiselFlatSpec {
  behavior of "FusionLoopMatmulGroupController"

  it should "gate VPU dispatch on all Gemmini child-stream completions" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/fusion-loop-matmul-group-control"
    )

    chisel3.iotesters.Driver.execute(args,
      () => new FusionLoopMatmulGroupController(nSharers = 2)) {
      c => new FusionLoopMatmulGroupControllerTester(c)
    } should be (true)
  }

  it should "support the singleton 1x16 fusion configuration" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/fusion-loop-matmul-group-singleton"
    )

    chisel3.iotesters.Driver.execute(args,
      () => new FusionLoopMatmulGroupController(nSharers = 1)) {
      c => new FusionLoopMatmulSingleMemberTester(c)
    } should be (true)
  }
}
