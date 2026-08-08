package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import Util._

class VsramAccessHarness(addressBits: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(new VsramAccess(addressBits))
    val b = Input(new VsramAccess(addressBits))
    val overlaps = Output(Bool())
    val conflicts = Output(Bool())
  })

  io.overlaps := io.a.overlaps(io.b)
  io.conflicts := io.a.conflicts(io.b)
}

class VsramAccessTester(c: VsramAccessHarness)
    extends PeekPokeTester(c) {
  private def drive(access: VsramAccess, start: Int, end: Int,
                    read: Boolean, write: Boolean,
                    wraps: Boolean = false): Unit = {
    poke(access.valid, 1)
    poke(access.start, start)
    poke(access.end, end)
    poke(access.wraps_around, wraps)
    poke(access.read, read)
    poke(access.write, write)
  }

  drive(c.io.a, 0, 4, read = false, write = true)
  drive(c.io.b, 4, 8, read = true, write = false)
  expect(c.io.overlaps, 0)
  expect(c.io.conflicts, 0)

  drive(c.io.b, 3, 8, read = true, write = false)
  expect(c.io.overlaps, 1)
  expect(c.io.conflicts, 1)

  drive(c.io.a, 6, 2, read = false, write = true, wraps = true)
  drive(c.io.b, 1, 3, read = true, write = false)
  expect(c.io.overlaps, 1)
  expect(c.io.conflicts, 1)

  drive(c.io.a, 0, 4, read = true, write = false)
  drive(c.io.b, 1, 3, read = true, write = false)
  expect(c.io.overlaps, 1)
  expect(c.io.conflicts, 0)
}

class VsramAccessUnitTest extends ChiselFlatSpec {
  behavior of "VsramAccess"

  it should "use half-open intervals and explicit read/write hazards" in {
    val testerArgs = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/vsram-access")
    chisel3.iotesters.Driver.execute(testerArgs, () =>
      new VsramAccessHarness(addressBits = 4)) { c =>
      new VsramAccessTester(c)
    } should be(true)
  }
}

class VsramExtEntriesTester(c: VsramExtEntries)
    extends PeekPokeTester(c) {
  private val ldq = 0
  private val exq = 1
  private val resMaxPerType = 2
  private val typeWidth = 1

  private def encodedId(queue: Int, localId: Int): Int =
    (queue << typeWidth) | localId

  private def clearAccess(access: VsramAccess): Unit = {
    poke(access.valid, 0)
    poke(access.start, 0)
    poke(access.end, 0)
    poke(access.wraps_around, 0)
    poke(access.read, 0)
    poke(access.write, 0)
  }

  private def driveAccess(access: VsramAccess, start: Int, end: Int,
                          read: Boolean, write: Boolean): Unit = {
    poke(access.valid, 1)
    poke(access.start, start)
    poke(access.end, end)
    poke(access.wraps_around, 0)
    poke(access.read, read)
    poke(access.write, write)
  }

  private def clearGemmini(sharer: Int): Unit = {
    val port = c.io.in(sharer)
    poke(port.alloc_entry.valid, 0)
    poke(port.alloc_entry.bits.alloc_id, 0)
    clearAccess(port.alloc_entry.bits.access)
    Seq(port.issue_ld, port.issue_ex, port.issue_st).foreach { issue =>
      poke(issue.valid, 0)
      poke(issue.bits.issue_id, 0)
      poke(issue.bits.valid, 0)
    }
    poke(port.complete_id.valid, 0)
    poke(port.complete_id.bits, 0)
  }

  private def driveGemminiAllocation(
      sharer: Int, queue: Int, localId: Int,
      start: Int, end: Int, read: Boolean, write: Boolean): Unit = {
    val port = c.io.in(sharer)
    poke(port.alloc_entry.valid, 1)
    poke(port.alloc_entry.bits.alloc_id, encodedId(queue, localId))
    driveAccess(port.alloc_entry.bits.access, start, end, read, write)
  }

  private def allocateGemmini(
      sharer: Int, queue: Int, localId: Int,
      start: Int, end: Int, read: Boolean, write: Boolean): Unit = {
    driveGemminiAllocation(sharer, queue, localId,
      start, end, read, write)
    step(1)
    clearGemmini(sharer)
  }

  private def completeGemmini(
      sharer: Int, queue: Int, localId: Int): Unit = {
    val port = c.io.in(sharer)
    poke(port.complete_id.valid, 1)
    poke(port.complete_id.bits, encodedId(queue, localId))
    step(1)
    poke(port.complete_id.valid, 0)
  }

  private def clearVpuAllocation(): Unit = {
    poke(c.io.vpu.allocate.valid, 0)
    poke(c.io.vpu.allocate.bits.slot, 0)
    c.io.vpu.allocate.bits.accesses.foreach(clearAccess)
  }

  private def driveVpuAllocation(
      slot: Int, start: Int, end: Int,
      read: Boolean, write: Boolean): Unit = {
    clearVpuAllocation()
    poke(c.io.vpu.allocate.valid, 1)
    poke(c.io.vpu.allocate.bits.slot, slot)
    driveAccess(c.io.vpu.allocate.bits.accesses(0),
      start, end, read, write)
  }

  private def allocateVpu(
      slot: Int, start: Int, end: Int,
      read: Boolean, write: Boolean): Unit = {
    driveVpuAllocation(slot, start, end, read, write)
    step(1)
    clearVpuAllocation()
  }

  private def releaseVpu(slots: Int*): Unit = {
    val mask = slots.foldLeft(BigInt(0)) {
      case (value, slot) => value | (BigInt(1) << slot)
    }
    poke(c.io.vpu.release, mask)
    step(1)
    poke(c.io.vpu.release, 0)
  }

  private def expectGemminiReady(
      sharer: Int, queue: Int, localId: Int, ready: Boolean): Unit = {
    val signal = if (queue == ldq) c.io.in(sharer).ld_deps_ready(localId)
      else c.io.in(sharer).ex_deps_ready(localId)
    expect(signal, if (ready) 1 else 0)
  }

  private def expectVpuReady(slot: Int, ready: Boolean): Unit =
    expect(c.io.vpu.ready(slot), if (ready) 1 else 0)

  c.io.in.indices.foreach(clearGemmini)
  clearVpuAllocation()
  poke(c.io.vpu.release, 0)
  step(1)

  // QK C_TO write -> VPU softmax read/write.
  allocateGemmini(0, exq, 0, 0, 4, read = false, write = true)
  allocateVpu(0, 0, 4, read = true, write = true)
  expectVpuReady(0, ready = false)
  completeGemmini(0, exq, 0)
  expectVpuReady(0, ready = true)

  // VPU P write -> PV A_FROM read.
  allocateGemmini(0, exq, 0, 0, 4, read = true, write = false)
  expectGemminiReady(0, exq, 0, ready = false)
  releaseVpu(0)
  expectGemminiReady(0, exq, 0, ready = true)
  completeGemmini(0, exq, 0)

  // One Gemmini command may depend on multiple live VPU writers.
  allocateVpu(0, 16, 20, read = false, write = true)
  allocateVpu(1, 16, 20, read = false, write = true)
  allocateGemmini(0, exq, 0, 16, 20, read = true, write = false)
  expectGemminiReady(0, exq, 0, ready = false)
  releaseVpu(0)
  expectGemminiReady(0, exq, 0, ready = false)
  releaseVpu(1)
  expectGemminiReady(0, exq, 0, ready = true)
  completeGemmini(0, exq, 0)

  // Disjoint ranges and read/read overlap do not serialize.
  allocateGemmini(0, exq, 0, 32, 36, read = false, write = true)
  allocateVpu(0, 40, 44, read = true, write = false)
  expectVpuReady(0, ready = true)
  completeGemmini(0, exq, 0)
  releaseVpu(0)

  allocateGemmini(0, exq, 0, 48, 52, read = true, write = false)
  allocateVpu(0, 48, 52, read = true, write = false)
  expectVpuReady(0, ready = true)
  completeGemmini(0, exq, 0)
  releaseVpu(0)

  // Same-cycle Gemmini and VPU allocation uses Gemmini-before-VPU age order.
  driveGemminiAllocation(1, ldq, 0, 64, 68,
    read = false, write = true)
  driveVpuAllocation(0, 64, 68, read = true, write = false)
  step(1)
  clearGemmini(1)
  clearVpuAllocation()
  expectVpuReady(0, ready = false)
  completeGemmini(1, ldq, 0)
  expectVpuReady(0, ready = true)
  releaseVpu(0)

  // Releasing and reusing one VPU slot in the same cycle must not transfer the
  // old occupant's dependency to the new occupant.
  allocateVpu(0, 80, 84, read = false, write = true)
  allocateGemmini(0, exq, 0, 80, 84, read = true, write = false)
  expectGemminiReady(0, exq, 0, ready = false)
  poke(c.io.vpu.release, 1)
  driveVpuAllocation(0, 80, 84, read = false, write = true)
  step(1)
  poke(c.io.vpu.release, 0)
  clearVpuAllocation()
  expectGemminiReady(0, exq, 0, ready = true)
  expectVpuReady(0, ready = false)
  completeGemmini(0, exq, 0)
  expectVpuReady(0, ready = true)
  releaseVpu(0)

  // STORE has no VSRAM fusion operation and is never blocked by this table.
  c.io.in.indices.foreach { sharer =>
    expect(c.io.in(sharer).st_deps_ready(0), 1)
  }
}

class VsramExtEntriesUnitTest extends ChiselFlatSpec {
  behavior of "VsramExtEntries"

  it should "track only cross Gemmini-VPU VSRAM hazards" in {
    val testerArgs = Array(
      "--backend-name", "treadle",
      "--target-dir", "test_run_dir/vsram-ext-entries")
    chisel3.iotesters.Driver.execute(testerArgs, () =>
      new VsramExtEntries(
        nSharers = 2,
        localAddr = new LocalAddr(8, 512, 8, 256),
        reservationStationEntriesLd = 2,
        reservationStationEntriesEx = 2,
        reservationStationEntriesSt = 1,
        resMaxPerType = 2,
        vpuEntries = 3)) { c =>
      new VsramExtEntriesTester(c)
    } should be(true)
  }
}
