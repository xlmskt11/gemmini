package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

class ExecuteCompletionCollisionHarness extends Module {
  val io = IO(new Bundle {
    val configEligible = Input(Bool())
    val fusionMeshCompletionDeqValid = Input(Bool())
    val configPop = Output(Bool())
  })

  io.configPop := ExecuteCompletionCollision.configMayPop(
    io.configEligible, io.fusionMeshCompletionDeqValid)
}

class ExecuteCompletionCollisionTester(c: ExecuteCompletionCollisionHarness)
    extends PeekPokeTester(c) {
  poke(c.io.configEligible, false)
  poke(c.io.fusionMeshCompletionDeqValid, false)
  expect(c.io.configPop, false)

  poke(c.io.configEligible, true)
  expect(c.io.configPop, true)

  // The fused completion owns ExecuteController's single-wide completion
  // port. The CONFIG command must remain at the command-queue head.
  poke(c.io.fusionMeshCompletionDeqValid, true)
  expect(c.io.configPop, false)
  step(1)

  // Keeping the CONFIG candidate asserted models the unpopped queue head. It
  // may pop as soon as the queued fused completion has drained.
  poke(c.io.fusionMeshCompletionDeqValid, false)
  expect(c.io.configPop, true)
}

class ExecuteCompletionCollisionUnitTest extends ChiselFlatSpec {
  behavior of "ExecuteController completion collision policy"

  it should "hold CONFIG while a fused mesh completion dequeues" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/execute-completion-collision"
    )

    chisel3.iotesters.Driver.execute(args,
      () => new ExecuteCompletionCollisionHarness) {
      c => new ExecuteCompletionCollisionTester(c)
    } should be (true)
  }
}
