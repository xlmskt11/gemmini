package gemmini

import chisel3._
import chisel3.util._

/** One raw VPU read from the unified Gemmini accumulator.
  *
  * `address` is the first physical-row element base. `fragmentStride` is the
  * element distance between the lower and upper lane fragments. It equals one
  * ACC row for ordinary vectors and the configured tile stride for segmented
  * vectors.
  */
class GemminiVpuMemoryReadRequest(
    elementAddrBits: Int,
    nLanes: Int,
    tagBits: Int) extends Bundle {
  val address = UInt(elementAddrBits.W)
  val fragmentStride = UInt(elementAddrBits.W)
  val laneMask = Vec(nLanes, Bool())
  val tag = UInt(tagBits.W)
}

class GemminiVpuMemoryReadResponse(
    nLanes: Int,
    elementBits: Int,
    tagBits: Int) extends Bundle {
  val data = Vec(nLanes, UInt(elementBits.W))
  val tag = UInt(tagBits.W)
}

/** One raw, masked VPU write to the unified Gemmini accumulator. */
class GemminiVpuMemoryWriteRequest(
    elementAddrBits: Int,
    nLanes: Int,
    elementBits: Int) extends Bundle {
  val address = UInt(elementAddrBits.W)
  val fragmentStride = UInt(elementAddrBits.W)
  val data = Vec(nLanes, UInt(elementBits.W))
  val laneMask = Vec(nLanes, Bool())
}

/**
  * VPU-client view of the unified accumulator interface.
  *
  * The fused VPU owns three independent read clients (source 0, source 1 and
  * store) and two independent write clients (execute and load). A backend uses
  * `Flipped(new GemminiVpuMemoryIO(...))`. A write completes when its request
  * fires; the memory blocks a same-address read while the physical write
  * remains pending.
  */
class GemminiVpuMemoryIO(
    elementAddrBits: Int,
    nLanes: Int,
    elementBits: Int,
    tagBits: Int,
    nReadClients: Int = 3,
    nWriteClients: Int = 2) extends Bundle {
  require(elementAddrBits > 0)
  require(nLanes > 0)
  require(elementBits > 0)
  require(tagBits > 0)
  require(nReadClients > 0)
  require(nWriteClients > 0)

  val readRequest = Vec(nReadClients, Decoupled(
    new GemminiVpuMemoryReadRequest(elementAddrBits, nLanes, tagBits)))
  val readResponse = Flipped(Vec(nReadClients, Decoupled(
    new GemminiVpuMemoryReadResponse(nLanes, elementBits, tagBits))))
  val writeRequest = Vec(nWriteClients, Decoupled(
    new GemminiVpuMemoryWriteRequest(
      elementAddrBits, nLanes, elementBits)))

  val busy = Input(Bool())
  val readConflictStall = Input(Bool())
  val writeConflictStall = Input(Bool())
}

/** Presents several private Gemmini accumulators as one VPU-visible address
  * space. The global element address is encoded as `owner || localAddress`;
  * every owner therefore has the same power-of-two capacity.
  *
  * Reads retain the selected owner until that client's response is consumed.
  * This matches the one-outstanding-response contract of
  * [[VpuSharedAccumulatorAdapter]], while still permitting a replacement read
  * to issue in the cycle in which the previous response is consumed. Writes
  * have no response and are routed combinationally.
  */
class PrivateVpuMemoryRouter(
    val elementsPerOwner: Int,
    val elementsPerRow: Int,
    val nOwners: Int = 4,
    val nLanes: Int = 16,
    val elementBits: Int = 32,
    val tagBits: Int = 1,
    val nReadClients: Int = 3,
    val nWriteClients: Int = 2) extends Module {
  require(elementsPerOwner > 0 && isPow2(elementsPerOwner),
    "private VPU owner capacity must be a power of two")
  require(nOwners > 1 && isPow2(nOwners),
    "private VPU fusion requires a power-of-two owner count")
  require(elementsPerRow > 0 && nLanes % elementsPerRow == 0,
    "VPU lanes must contain an integral number of ACC rows")
  require(nReadClients > 0 && nWriteClients > 0)

  val localElementAddrBits: Int = log2Ceil(elementsPerOwner)
  val ownerBits: Int = log2Ceil(nOwners)
  val globalElementAddrBits: Int = localElementAddrBits + ownerBits
  private val fragments = nLanes / elementsPerRow

  val io = IO(new Bundle {
    val vpu = Flipped(new GemminiVpuMemoryIO(
      globalElementAddrBits, nLanes, elementBits, tagBits,
      nReadClients, nWriteClients))
    val owners = Vec(nOwners, new GemminiVpuMemoryIO(
      localElementAddrBits, nLanes, elementBits, tagBits,
      nReadClients, nWriteClients))
  })

  private def owner(address: UInt): UInt =
    address(globalElementAddrBits - 1, localElementAddrBits)

  private def localAddress(address: UInt): UInt =
    address(localElementAddrBits - 1, 0)

  private def fragmentActive(mask: Vec[Bool], fragment: Int): Bool = {
    val first = fragment * elementsPerRow
    mask.slice(first, first + elementsPerRow).reduce(_ || _)
  }

  /** A request is never allowed to make one architectural VPU word span two
    * private accumulators. Also reject wraparound at the top of the aggregate
    * address space, which could otherwise make the owner bits appear equal.
    */
  private def assertFragmentsStayWithinOwner(
      valid: Bool, address: UInt, fragmentStride: UInt,
      laneMask: Vec[Bool], operation: String): Unit = {
    val requestOwner = owner(address)
    for (fragment <- 0 until fragments) {
      val fragmentAddress = address +& fragmentStride * fragment.U
      when(valid && fragmentActive(laneMask, fragment)) {
        assert(!fragmentAddress(globalElementAddrBits),
          s"private VPU $operation fragment wrapped the aggregate ACC space")
        assert(fragmentAddress(globalElementAddrBits - 1,
          localElementAddrBits) === requestOwner,
          s"private VPU $operation fragments crossed an ACC owner boundary")
      }
    }
  }

  for (endpoint <- io.owners) {
    for (client <- 0 until nReadClients) {
      endpoint.readRequest(client).valid := false.B
      endpoint.readRequest(client).bits := DontCare
      endpoint.readResponse(client).ready := false.B
    }
    for (client <- 0 until nWriteClients) {
      endpoint.writeRequest(client).valid := false.B
      endpoint.writeRequest(client).bits := DontCare
    }
  }

  for (client <- 0 until nReadClients) {
    val request = io.vpu.readRequest(client)
    val requestOwner = owner(request.bits.address)
    val requestOwnerOH = UIntToOH(requestOwner, nOwners)
    val responsePending = RegInit(false.B)
    val responseOwner = RegInit(0.U(ownerBits.W))
    val responseOwnerOH = UIntToOH(responseOwner, nOwners)

    val selectedResponseValid = Mux1H(responseOwnerOH,
      io.owners.map(_.readResponse(client).valid))
    val selectedResponseBits = Mux1H(responseOwnerOH,
      io.owners.map(_.readResponse(client).bits))

    io.vpu.readResponse(client).valid :=
      responsePending && selectedResponseValid
    io.vpu.readResponse(client).bits := selectedResponseBits

    for (endpoint <- 0 until nOwners) {
      val endpointRequest = io.owners(endpoint).readRequest(client)
      // A flow-through response queue can retire the old request and admit a
      // replacement from this client in the same cycle.
      val canIssue = !responsePending || io.vpu.readResponse(client).fire
      endpointRequest.valid := request.valid && canIssue &&
        requestOwnerOH(endpoint)
      endpointRequest.bits.address := localAddress(request.bits.address)
      endpointRequest.bits.fragmentStride :=
        localAddress(request.bits.fragmentStride)
      endpointRequest.bits.laneMask := request.bits.laneMask
      endpointRequest.bits.tag := request.bits.tag
      io.owners(endpoint).readResponse(client).ready :=
        responsePending && responseOwnerOH(endpoint) &&
          io.vpu.readResponse(client).ready
    }

    val selectedRequestReady = Mux1H(requestOwnerOH,
      io.owners.map(_.readRequest(client).ready))
    request.ready := (!responsePending ||
      io.vpu.readResponse(client).fire) && selectedRequestReady

    when(request.fire) {
      responsePending := true.B
      responseOwner := requestOwner
    }.elsewhen(io.vpu.readResponse(client).fire) {
      responsePending := false.B
    }

    val unexpectedResponses = VecInit(io.owners.zipWithIndex.map {
      case (endpoint, index) =>
        endpoint.readResponse(client).valid &&
          (!responsePending || !responseOwnerOH(index))
    })
    assert(!unexpectedResponses.asUInt.orR,
      "private VPU ACC endpoint returned a response without a matching read")

    assertFragmentsStayWithinOwner(request.valid, request.bits.address,
      request.bits.fragmentStride, request.bits.laneMask, "read")
  }

  for (client <- 0 until nWriteClients) {
    val request = io.vpu.writeRequest(client)
    val requestOwner = owner(request.bits.address)
    val requestOwnerOH = UIntToOH(requestOwner, nOwners)

    for (endpoint <- 0 until nOwners) {
      val endpointRequest = io.owners(endpoint).writeRequest(client)
      endpointRequest.valid := request.valid && requestOwnerOH(endpoint)
      endpointRequest.bits.address := localAddress(request.bits.address)
      endpointRequest.bits.fragmentStride :=
        localAddress(request.bits.fragmentStride)
      endpointRequest.bits.data := request.bits.data
      endpointRequest.bits.laneMask := request.bits.laneMask
    }
    request.ready := Mux1H(requestOwnerOH,
      io.owners.map(_.writeRequest(client).ready))

    assertFragmentsStayWithinOwner(request.valid, request.bits.address,
      request.bits.fragmentStride, request.bits.laneMask, "write")
  }

  io.vpu.busy := VecInit(io.owners.map(_.busy)).asUInt.orR
  io.vpu.readConflictStall :=
    VecInit(io.owners.map(_.readConflictStall)).asUInt.orR
  io.vpu.writeConflictStall :=
    VecInit(io.owners.map(_.writeConflictStall)).asUInt.orR
}
