package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import VpuLocalAddr._

class FusionLocalAddrFlagHarness extends Module {
  private val localAddrType = new LocalAddr(
    sp_banks = 8, sp_bank_entries = 256,
    acc_banks = 8, acc_bank_entries = 256)

  val io = IO(new Bundle {
    val base = Input(UInt(11.W))
    val a_from_vsram = Input(Bool())
    val c_to_vsram = Input(Bool())
    val d_from_vsram = Input(Bool())
    val out_data = Output(UInt(11.W))
    val a_flag = Output(Bool())
    val c_flag = Output(Bool())
    val d_flag = Output(Bool())
  })

  val baseAddr = LocalAddr.cast_to_sp_addr(localAddrType, io.base)
  val withA = with_a_from_vsram(baseAddr, io.a_from_vsram)
  val withBoth = with_c_to_vsram(withA, io.c_to_vsram)
  val withAll = with_d_from_vsram(withBoth, io.d_from_vsram)

  io.out_data := withAll.full_sp_addr()
  io.a_flag := withAll.a_from_vsram()
  io.c_flag := withAll.c_to_vsram()
  io.d_flag := withAll.d_from_vsram()
}

class FusionLocalAddrFlagTester(c: FusionLocalAddrFlagHarness) extends PeekPokeTester(c) {
  for {
    a <- Seq(false, true)
    cFlag <- Seq(false, true)
    d <- Seq(false, true)
  } {
    poke(c.io.base, 0x345)
    poke(c.io.a_from_vsram, a)
    poke(c.io.c_to_vsram, cFlag)
    poke(c.io.d_from_vsram, d)
    step(1)
    expect(c.io.out_data, 0x345)
    expect(c.io.a_flag, a)
    expect(c.io.c_flag, cFlag)
    expect(c.io.d_flag, d)
  }
}

class LdBCompleteControlTester(c: LdBCompleteControl) extends PeekPokeTester(c) {
  private val nSharers = 4
  private val gemv = c.io.gemv.get

  private def driveIdleMember(i: Int): Unit = {
    poke(c.io.in(i).ldb.group_id, 0)
    poke(c.io.in(i).ldb.group_list, 0)
    poke(c.io.in(i).ldb.has_gemv_followup.get, false)
    poke(c.io.in(i).ldb.k, 0)
    poke(c.io.in(i).ldb.k_offset, 0)
    poke(c.io.in(i).ldb.max_k, 0)
    poke(c.io.in(i).ldb.j, 0)
    poke(c.io.in(i).ldb.idle, true)

    poke(c.io.in(i).ex.group_id, 0)
    poke(c.io.in(i).ex.group_list, 0)
    poke(c.io.in(i).ex.k, 0)
    poke(c.io.in(i).ex.j, 0)
    poke(c.io.in(i).ex.idle, true)

    poke(c.io.in(i).stc.group_id, 0)
    poke(c.io.in(i).stc.idle, true)
  }

  private def driveActiveMember(i: Int, groupId: Int, groupList: Int,
                                hasGemv: Boolean, idle: Boolean): Unit = {
    poke(c.io.in(i).ldb.group_id, groupId)
    poke(c.io.in(i).ldb.group_list, groupList)
    poke(c.io.in(i).ldb.has_gemv_followup.get, hasGemv)
    poke(c.io.in(i).ldb.k, 0)
    poke(c.io.in(i).ldb.k_offset, 0)
    poke(c.io.in(i).ldb.max_k, 1)
    poke(c.io.in(i).ldb.j, 0)
    poke(c.io.in(i).ldb.idle, idle)

    poke(c.io.in(i).ex.group_id, groupId)
    poke(c.io.in(i).ex.group_list, groupList)
    poke(c.io.in(i).ex.k, 0)
    poke(c.io.in(i).ex.j, 0)
    poke(c.io.in(i).ex.idle, idle)

    poke(c.io.in(i).stc.group_id, groupId)
    poke(c.io.in(i).stc.idle, idle)
  }

  private def driveMemberIdleState(i: Int, ldbIdle: Boolean,
                                   exIdle: Boolean, stcIdle: Boolean): Unit = {
    poke(c.io.in(i).ldb.idle, ldbIdle)
    poke(c.io.in(i).ex.idle, exIdle)
    poke(c.io.in(i).stc.idle, stcIdle)
  }

  private def driveGemv(pending: Boolean, groupId: Int,
                        lastDispatchFire: Boolean = false, abort: Boolean = false): Unit = {
    poke(gemv.pending, pending)
    poke(gemv.group_id, groupId)
    poke(gemv.last_dispatch_fire, lastDispatchFire)
    poke(gemv.abort, abort)
  }

  for (i <- 0 until nSharers) driveIdleMember(i)
  driveGemv(pending = true, groupId = 0)
  expect(gemv.group_allocated, false)
  expect(gemv.dispatch_enable, false)
  expect(gemv.dispatch_reject, false)

  // Allocate a two-member group which requires a VPU follow-up. Both members
  // advertise the same bit, exercising the agreement check as well.
  driveActiveMember(0, groupId = 0, groupList = 0x3, hasGemv = true, idle = false)
  driveActiveMember(1, groupId = 0, groupList = 0x3, hasGemv = true, idle = false)
  step(1)

  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, false)
  expect(gemv.dispatch_reject, false)

  // Each existing completion edge represents the corresponding child stream
  // entering the local RS, SharedExtEntries, and VsramExtEntries. No partial conjunction
  // may open VPU dispatch.
  driveMemberIdleState(0, ldbIdle = true, exIdle = false, stcIdle = false)
  driveMemberIdleState(1, ldbIdle = true, exIdle = false, stcIdle = false)
  step(1)
  expect(gemv.dispatch_enable, false)

  driveMemberIdleState(0, ldbIdle = true, exIdle = true, stcIdle = false)
  driveMemberIdleState(1, ldbIdle = true, exIdle = true, stcIdle = false)
  step(1)
  expect(gemv.dispatch_enable, false)

  // Dispatch opens only after all three streams are registered. These are
  // unroller-completion signals, not functional-unit retirement signals.
  driveMemberIdleState(0, ldbIdle = true, exIdle = true, stcIdle = true)
  driveMemberIdleState(1, ldbIdle = true, exIdle = true, stcIdle = true)
  step(1)

  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, true)

  // The final VPU command entering its RS/SharedDeps closes the coarse group.
  // Any still-running Gemmini/VPU commands retain their address dependencies
  // in the dependency tables, so the finite group ID is immediately reusable.
  driveGemv(pending = true, groupId = 0, lastDispatchFire = true)
  step(1)
  driveGemv(pending = true, groupId = 0)
  expect(gemv.group_allocated, false)
  expect(gemv.dispatch_enable, false)

  // Reuse the released finite-width ID for an independent group. No stale
  // registration or VPU-dispatch state may survive from the previous owner.
  driveActiveMember(0, groupId = 0, groupList = 0x1, hasGemv = true,
    idle = false)
  for (i <- 1 until nSharers) driveIdleMember(i)
  expect(c.io.in(0).loop_full, true)
  step(1)
  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, false)

  driveActiveMember(0, groupId = 0, groupList = 0x1, hasGemv = true,
    idle = true)
  step(1)
  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, true)

  driveGemv(pending = true, groupId = 0, lastDispatchFire = true)
  step(1)
  expect(gemv.group_allocated, false)
  expect(gemv.dispatch_enable, false)

  // A legacy group with HAS_GEMV_FOLLOWUP clear releases after the ordinary
  // Gemmini completion path without waiting for any VPU handshake.
  driveActiveMember(0, groupId = 1, groupList = 0x1, hasGemv = false, idle = false)
  driveGemv(pending = true, groupId = 1)
  step(1)
  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, false)
  expect(gemv.dispatch_reject, true)

  driveActiveMember(0, groupId = 1, groupList = 0x1, hasGemv = false,
    idle = true)
  step(1)
  expect(gemv.group_allocated, true)
  expect(gemv.dispatch_enable, false)

  step(1)
  expect(gemv.group_allocated, false)

  // Abort follows the same registered release rule as a successful final
  // dispatch, while still waiting for the Gemmini side to be complete.
  driveActiveMember(0, groupId = 2, groupList = 0x1, hasGemv = true, idle = false)
  driveGemv(pending = true, groupId = 2)
  step(1)
  driveActiveMember(0, groupId = 2, groupList = 0x1, hasGemv = true,
    idle = true)
  step(1)
  expect(gemv.dispatch_enable, true)

  driveGemv(pending = true, groupId = 2, abort = true)
  step(1)
  expect(gemv.group_allocated, false)
  expect(gemv.dispatch_enable, false)
}

class LdBCompleteControlUnitTest extends ChiselFlatSpec {
  behavior of "LdBCompleteControl"

  it should "hold a Gemmini group until the final grouped VPU command is dispatched" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/ldb-complete-control"
    )

    chisel3.iotesters.Driver.execute(args, () => new LdBCompleteControl(
      nSharers = 4, useVpuFusion = true)) {
      c => new LdBCompleteControlTester(c)
    } should be (true)
  }

  it should "encode fusion routing flags without changing the LocalAddr data field" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/fusion-local-addr-flags"
    )

    chisel3.iotesters.Driver.execute(args, () => new FusionLocalAddrFlagHarness) {
      c => new FusionLocalAddrFlagTester(c)
    } should be (true)
  }

}
