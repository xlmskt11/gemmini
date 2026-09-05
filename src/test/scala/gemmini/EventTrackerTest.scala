package gemmini

import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

class EventTrackerTester(c: EventTracker) extends PeekPokeTester(c) {
  private val nAdmissions = 4
  private val nCompletions = 3

  private def clearAdmission(portId: Int): Unit = {
    poke(c.io.admissions(portId).request, false)
    poke(c.io.admissions(portId).waitValid, false)
    poke(c.io.admissions(portId).waitId, 0)
    poke(c.io.admissions(portId).produceValid, false)
    poke(c.io.admissions(portId).produceId, 0)
    poke(c.io.admissions(portId).produceSeal, false)
    poke(c.io.admissions(portId).commit, false)
  }

  private def clearCompletion(portId: Int): Unit = {
    poke(c.io.completions(portId).valid, false)
    poke(c.io.completions(portId).id, 0)
  }

  private def clearInputs(): Unit = {
    for (portId <- 0 until nAdmissions) clearAdmission(portId)
    for (portId <- 0 until nCompletions) clearCompletion(portId)
  }

  private def produce(
    portId: Int,
    eventId: Int,
    seal: Boolean,
    commit: Boolean = false
  ): Unit = {
    clearAdmission(portId)
    poke(c.io.admissions(portId).request, true)
    poke(c.io.admissions(portId).produceValid, true)
    poke(c.io.admissions(portId).produceId, eventId)
    poke(c.io.admissions(portId).produceSeal, seal)
    poke(c.io.admissions(portId).commit, commit)
  }

  private def waitFor(
    portId: Int,
    eventId: Int,
    commit: Boolean = false
  ): Unit = {
    clearAdmission(portId)
    poke(c.io.admissions(portId).request, true)
    poke(c.io.admissions(portId).waitValid, true)
    poke(c.io.admissions(portId).waitId, eventId)
    poke(c.io.admissions(portId).commit, commit)
  }

  private def complete(portId: Int, eventId: Int): Unit = {
    poke(c.io.completions(portId).valid, true)
    poke(c.io.completions(portId).id, eventId)
  }

  clearInputs()
  step(1)

  // Two producer slots may attach to the same unused event in one cycle. The
  // sealing attachment closes it to later producers, but the wait remains
  // blocked until both attachments complete.
  produce(portId = 0, eventId = 3, seal = false)
  produce(portId = 1, eventId = 3, seal = true)
  expect(c.io.admissions(0).ready, true)
  expect(c.io.admissions(1).ready, true)
  poke(c.io.admissions(0).commit, true)
  poke(c.io.admissions(1).commit, true)
  step(1)

  clearInputs()
  waitFor(portId = 2, eventId = 3)
  expect(c.io.admissions(2).ready, false)

  complete(portId = 0, eventId = 3)
  step(1)
  clearCompletion(0)
  expect(c.io.admissions(2).ready, false)

  complete(portId = 1, eventId = 3)
  step(1)
  clearCompletion(1)
  expect(c.io.admissions(2).ready, true)

  // Ready is sticky, and all current requesters for the same event see the
  // same collective readiness.
  waitFor(portId = 3, eventId = 3)
  expect(c.io.admissions(2).ready, true)
  expect(c.io.admissions(3).ready, true)
  step(2)
  expect(c.io.admissions(2).ready, true)
  expect(c.io.admissions(3).ready, true)

  // A sealed event cannot be recycled by a producer in its consume cycle.
  produce(portId = 0, eventId = 3, seal = true)
  expect(c.io.admissions(0).ready, false)
  expect(c.io.admissions(2).ready, true)
  expect(c.io.admissions(3).ready, true)
  poke(c.io.admissions(2).commit, true)
  poke(c.io.admissions(3).commit, true)
  step(1)

  clearInputs()
  waitFor(portId = 2, eventId = 3)
  expect(c.io.admissions(2).ready, false)

  // Once the collective consumer has freed the ID, it can be armed again on
  // the next cycle.
  clearAdmission(2)
  produce(portId = 0, eventId = 3, seal = true)
  expect(c.io.admissions(0).ready, true)
  poke(c.io.admissions(0).commit, true)
  step(1)
  clearInputs()
  complete(portId = 0, eventId = 3)
  step(1)
  clearInputs()
  waitFor(portId = 0, eventId = 3)
  expect(c.io.admissions(0).ready, true)
  poke(c.io.admissions(0).commit, true)
  step(1)

  // An unsealed event can temporarily reach pending=0 and accept a later
  // producer. Multiple completions in one cycle exercise completion PopCount.
  clearInputs()
  produce(portId = 0, eventId = 5, seal = false)
  produce(portId = 1, eventId = 5, seal = false)
  poke(c.io.admissions(0).commit, true)
  poke(c.io.admissions(1).commit, true)
  step(1)

  clearInputs()
  complete(portId = 0, eventId = 5)
  complete(portId = 1, eventId = 5)
  step(1)

  clearInputs()
  waitFor(portId = 3, eventId = 5)
  expect(c.io.admissions(3).ready, false)
  produce(portId = 0, eventId = 5, seal = true)
  expect(c.io.admissions(0).ready, true)
  poke(c.io.admissions(0).commit, true)
  step(1)

  clearInputs()
  waitFor(portId = 3, eventId = 5)
  expect(c.io.admissions(3).ready, false)
  complete(portId = 2, eventId = 5)
  step(1)
  clearCompletion(2)
  expect(c.io.admissions(3).ready, true)

  // A one-command producer may attach and complete in the same cycle.
  clearInputs()
  produce(portId = 0, eventId = 6, seal = true, commit = true)
  complete(portId = 0, eventId = 6)
  expect(c.io.admissions(0).ready, true)
  step(1)
  clearInputs()
  waitFor(portId = 1, eventId = 6)
  expect(c.io.admissions(1).ready, true)
}

class EventTrackerUnitTest extends ChiselFlatSpec {
  behavior of "EventTracker"

  it should "track multi-port producers and collective consumers" in {
    val args = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/event-tracker"
    )

    chisel3.iotesters.Driver.execute(args,
      () => new EventTracker(nAdmissions = 4, nCompletions = 3)) {
      c => new EventTrackerTester(c)
    } should be (true)
  }
}
