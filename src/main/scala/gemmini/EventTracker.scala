package gemmini

import chisel3._
import chisel3.util.PopCount

object EventTracker {
  val Entries = 8
  val IdWidth = 3
}

/** One command-admission view of the event table.
  *
  * `request` presents the command at an admission point. `ready` says that
  * both its optional wait and producer attachment can be accepted. `commit`
  * is asserted only when that command is actually admitted.
  */
class EventTrackerAdmissionPort extends Bundle {
  val request = Input(Bool())

  val waitValid = Input(Bool())
  val waitId = Input(UInt(EventTracker.IdWidth.W))

  val produceValid = Input(Bool())
  val produceId = Input(UInt(EventTracker.IdWidth.W))
  val produceSeal = Input(Bool())

  val ready = Output(Bool())
  val commit = Input(Bool())
}

/** Releases one producer attachment after its command stream has been
  * registered.
  */
class EventTrackerCompletionPort extends Bundle {
  val valid = Input(Bool())
  val id = Input(UInt(EventTracker.IdWidth.W))
}

/** Tracks command-registration events independently of execution hazards.
  *
  * An event is armed by its first producer attachment. Every committed
  * producer attachment increments `pending`, and every completion decrements
  * it. The final producer attachment seals the event. Consumers may commit
  * once the event is valid, sealed, and has no pending producers; their commit
  * consumes and frees the entry.
  */
class EventTracker(
  nAdmissions: Int,
  nCompletions: Int,
  pendingWidth: Int = 4
) extends Module {
  require(nAdmissions > 0)
  require(nCompletions > 0)
  require(pendingWidth > 0)

  val io = IO(new Bundle {
    val admissions = Vec(nAdmissions, new EventTrackerAdmissionPort)
    val completions = Vec(nCompletions, new EventTrackerCompletionPort)
  })

  private val valid = RegInit(VecInit(Seq.fill(EventTracker.Entries)(false.B)))
  private val sealedEvent = RegInit(VecInit(Seq.fill(EventTracker.Entries)(false.B)))
  private val pending = RegInit(VecInit(Seq.fill(EventTracker.Entries)(0.U(pendingWidth.W))))

  // A producer may arm an unused entry or join an existing, unsealed event.
  // A sealed entry cannot be recycled in its consume cycle.
  private val producerAvailable = Wire(Vec(nAdmissions, Bool()))
  for (portId <- 0 until nAdmissions) {
    val port = io.admissions(portId)
    producerAvailable(portId) := !port.produceValid ||
      !valid(port.produceId) || !sealedEvent(port.produceId)
  }

  // All requesters waiting on one event form one collective consumer. Their
  // readiness includes every member's producer-side constraint, so matching
  // requesters always observe the same ready value.
  for (portId <- 0 until nAdmissions) {
    val port = io.admissions(portId)
    val waitReady = valid(port.waitId) && sealedEvent(port.waitId) &&
      (pending(port.waitId) === 0.U)
    val collectiveProducerReady = (0 until nAdmissions).map { peerId =>
      val peer = io.admissions(peerId)
      val matchingWaiter = peer.request && peer.waitValid &&
        peer.waitId === port.waitId
      !matchingWaiter || producerAvailable(peerId)
    }.reduce(_ && _)

    port.ready := port.request && producerAvailable(portId) &&
      (!port.waitValid || (waitReady && collectiveProducerReady))

    when(port.commit) {
      assert(port.request, "EventTracker commit without request")
      assert(port.ready, "EventTracker commit while admission is not ready")
    }
    when(port.commit && port.produceValid) {
      assert(!(valid(port.produceId) && sealedEvent(port.produceId)),
        "EventTracker producer attached to a sealed event")
    }
  }

  for (eventId <- 0 until EventTracker.Entries) {
    val waitRequests = VecInit((0 until nAdmissions).map { portId =>
      val port = io.admissions(portId)
      port.request && port.waitValid && port.waitId === eventId.U
    })
    val waitCommits = VecInit((0 until nAdmissions).map { portId =>
      val port = io.admissions(portId)
      port.commit && port.request && port.waitValid &&
        port.waitId === eventId.U
    })
    val consume = waitCommits.asUInt.orR

    // A collective wait either remains presented or commits in full. The
    // entry itself is consumed only once regardless of requester count.
    when(consume) {
      assert(waitCommits.asUInt === waitRequests.asUInt,
        "EventTracker collective consumers must commit together")
    }

    val attaches = VecInit((0 until nAdmissions).map { portId =>
      val port = io.admissions(portId)
      port.commit && port.request && port.produceValid &&
        port.produceId === eventId.U
    })
    val seals = VecInit((0 until nAdmissions).map { portId =>
      val port = io.admissions(portId)
      attaches(portId) && port.produceSeal
    })
    val completions = VecInit((0 until nCompletions).map { portId =>
      val port = io.completions(portId)
      port.valid && port.id === eventId.U
    })

    val attachCount = PopCount(attaches)
    val completionCount = PopCount(completions)
    val hasAttach = attaches.asUInt.orR
    val hasSeal = seals.asUInt.orR
    val hasCompletion = completions.asUInt.orR

    when(hasAttach) {
      assert(!consume, "EventTracker cannot recycle an event in its consume cycle")
    }

    val pendingWithAttach = pending(eventId) +& attachCount
    when(hasCompletion) {
      assert(valid(eventId) || hasAttach,
        "EventTracker completion for an invalid event")
      assert(completionCount <= pendingWithAttach,
        "EventTracker pending count underflow")
    }
    assert(pendingWithAttach <= ((BigInt(1) << pendingWidth) - 1).U,
      "EventTracker pending count overflow")
    val pendingAfterCompletion = pendingWithAttach - completionCount

    when(consume) {
      valid(eventId) := false.B
      sealedEvent(eventId) := false.B
      pending(eventId) := 0.U
    }.otherwise {
      when(hasAttach) {
        valid(eventId) := true.B
      }
      when(hasSeal) {
        sealedEvent(eventId) := true.B
      }
      when(hasAttach || hasCompletion) {
        pending(eventId) := pendingAfterCompletion
      }
    }
  }

}
