package gemmini

import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}

class PrivateVpuMemoryRouterTester(c: PrivateVpuMemoryRouter)
    extends PeekPokeTester(c) {
  private val nOwners = 4
  private val nReadClients = 3
  private val nWriteClients = 2
  private val nLanes = 16

  for (client <- 0 until nReadClients) {
    poke(c.io.vpu.readRequest(client).valid, false)
    poke(c.io.vpu.readRequest(client).bits.address, 0)
    poke(c.io.vpu.readRequest(client).bits.fragmentStride, 8)
    poke(c.io.vpu.readRequest(client).bits.tag, 0)
    c.io.vpu.readRequest(client).bits.laneMask.foreach(poke(_, false))
    poke(c.io.vpu.readResponse(client).ready, false)
  }
  for (client <- 0 until nWriteClients) {
    poke(c.io.vpu.writeRequest(client).valid, false)
    poke(c.io.vpu.writeRequest(client).bits.address, 0)
    poke(c.io.vpu.writeRequest(client).bits.fragmentStride, 8)
    c.io.vpu.writeRequest(client).bits.data.foreach(poke(_, 0))
    c.io.vpu.writeRequest(client).bits.laneMask.foreach(poke(_, false))
  }
  for (endpoint <- 0 until nOwners) {
    poke(c.io.owners(endpoint).busy, false)
    poke(c.io.owners(endpoint).readConflictStall, false)
    poke(c.io.owners(endpoint).writeConflictStall, false)
    for (client <- 0 until nReadClients) {
      poke(c.io.owners(endpoint).readRequest(client).ready, false)
      poke(c.io.owners(endpoint).readResponse(client).valid, false)
      poke(c.io.owners(endpoint).readResponse(client).bits.tag, 0)
      c.io.owners(endpoint).readResponse(client).bits.data.foreach(
        poke(_, 0))
    }
    for (client <- 0 until nWriteClients) {
      poke(c.io.owners(endpoint).writeRequest(client).ready, false)
    }
  }
  step(2)

  val fullMask = (0 until nLanes).toSet
  def setRead(owner: Int, local: Int, tag: Int): Unit = {
    poke(c.io.vpu.readRequest(0).valid, true)
    poke(c.io.vpu.readRequest(0).bits.address, owner * 64 + local)
    poke(c.io.vpu.readRequest(0).bits.fragmentStride, 8)
    poke(c.io.vpu.readRequest(0).bits.tag, tag)
    for (lane <- 0 until nLanes) {
      poke(c.io.vpu.readRequest(0).bits.laneMask(lane),
        fullMask.contains(lane))
    }
  }

  // Global owner 2 becomes local address 16. No other endpoint observes the
  // request.
  setRead(owner = 2, local = 16, tag = 0x91)
  poke(c.io.owners(2).readRequest(0).ready, true)
  expect(c.io.vpu.readRequest(0).ready, true)
  for (endpoint <- 0 until nOwners) {
    expect(c.io.owners(endpoint).readRequest(0).valid, endpoint == 2)
  }
  expect(c.io.owners(2).readRequest(0).bits.address, 16)
  expect(c.io.owners(2).readRequest(0).bits.fragmentStride, 8)
  expect(c.io.owners(2).readRequest(0).bits.tag, 0x91)
  step(1)

  poke(c.io.vpu.readRequest(0).valid, false)
  poke(c.io.owners(2).readRequest(0).ready, false)
  poke(c.io.owners(2).readResponse(0).valid, true)
  poke(c.io.owners(2).readResponse(0).bits.tag, 0x91)
  for (lane <- 0 until nLanes) {
    poke(c.io.owners(2).readResponse(0).bits.data(lane), 0x100 + lane)
  }
  expect(c.io.vpu.readResponse(0).valid, true)
  expect(c.io.vpu.readResponse(0).bits.tag, 0x91)

  // One response is outstanding per read client, so owner 1 remains blocked
  // until owner 2's result is consumed.
  setRead(owner = 1, local = 24, tag = 0x52)
  poke(c.io.owners(1).readRequest(0).ready, true)
  expect(c.io.vpu.readRequest(0).ready, false)
  expect(c.io.owners(1).readRequest(0).valid, false)

  // Consuming the old result admits the replacement request in this cycle.
  poke(c.io.vpu.readResponse(0).ready, true)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(c.io.owners(1).readRequest(0).valid, true)
  expect(c.io.owners(1).readRequest(0).bits.address, 24)
  step(1)

  poke(c.io.owners(2).readResponse(0).valid, false)
  poke(c.io.vpu.readResponse(0).ready, false)
  poke(c.io.vpu.readRequest(0).valid, false)
  poke(c.io.owners(1).readRequest(0).ready, false)
  poke(c.io.owners(1).readResponse(0).valid, true)
  poke(c.io.owners(1).readResponse(0).bits.tag, 0x52)
  for (lane <- 0 until nLanes) {
    poke(c.io.owners(1).readResponse(0).bits.data(lane), 0x200 + lane)
  }
  expect(c.io.vpu.readResponse(0).valid, true)
  expect(c.io.vpu.readResponse(0).bits.tag, 0x52)
  poke(c.io.vpu.readResponse(0).ready, true)
  step(1)
  poke(c.io.owners(1).readResponse(0).valid, false)
  poke(c.io.vpu.readResponse(0).ready, false)

  // Writes route without an owner-response register.
  val writeOwner = 3
  val writeLocal = 24
  poke(c.io.vpu.writeRequest(1).valid, true)
  poke(c.io.vpu.writeRequest(1).bits.address,
    writeOwner * 64 + writeLocal)
  poke(c.io.vpu.writeRequest(1).bits.fragmentStride, 8)
  poke(c.io.owners(writeOwner).writeRequest(1).ready, true)
  for (lane <- 0 until nLanes) {
    poke(c.io.vpu.writeRequest(1).bits.laneMask(lane), true)
    poke(c.io.vpu.writeRequest(1).bits.data(lane), 0x300 + lane)
  }
  expect(c.io.vpu.writeRequest(1).ready, true)
  for (endpoint <- 0 until nOwners) {
    expect(c.io.owners(endpoint).writeRequest(1).valid,
      endpoint == writeOwner)
  }
  expect(c.io.owners(writeOwner).writeRequest(1).bits.address, writeLocal)
  expect(c.io.owners(writeOwner).writeRequest(1).bits.fragmentStride, 8)
  step(1)
  poke(c.io.vpu.writeRequest(1).valid, false)

  // Status sidebands aggregate across all private endpoints.
  poke(c.io.owners(0).busy, true)
  poke(c.io.owners(1).readConflictStall, true)
  poke(c.io.owners(3).writeConflictStall, true)
  expect(c.io.vpu.busy, true)
  expect(c.io.vpu.readConflictStall, true)
  expect(c.io.vpu.writeConflictStall, true)
}

class PrivateVpuMemoryRouterUnitTest extends ChiselFlatSpec {
  behavior of "PrivateVpuMemoryRouter"

  it should "route owner-local requests and retain each read owner" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/private-vpu-memory-router")
    chisel3.iotesters.Driver.execute(args, () =>
      new PrivateVpuMemoryRouter(
        elementsPerOwner = 64,
        elementsPerRow = 8,
        nOwners = 4,
        nLanes = 16,
        elementBits = 32,
        tagBits = 8)) { c =>
      new PrivateVpuMemoryRouterTester(c)
    } should be(true)
  }
}
