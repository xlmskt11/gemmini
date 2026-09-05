package gemmini

import chisel3._
import chisel3.util._

/** Adapts the fused VPU's 16-lane raw memory requests to Gemmini's physical
  * accumulator banks. The VPU address is an FP32 element address, while each
  * Gemmini ACC request addresses one `blockCols`-element row.
  *
  * For an 8-column array, one VPU request is split into two rows separated by
  * the request's fragment stride. Distinct physical sub-banks accept both
  * fragments together; a stride which aliases one sub-bank is handled in two
  * cycles without reducing the architectural VL. A 16-column array uses one
  * row and follows the same path. Reads bypass scaling and activation. Writes
  * complete when the shared ACC accepts them; its pending-write address check
  * blocks a same-address read until the physical write pipeline drains.
  */
class VpuSharedAccumulatorAdapter[T <: Data, U <: Data, V <: Data](
    config: GemminiArrayConfig[T, U, V],
    vpuLanes: Int,
    tagBits: Int,
    nReadClients: Int = 3,
    nWriteClients: Int = 2)(implicit ev: Arithmetic[T]) extends Module {
  import config._

  private val blockCols = meshColumns * tileColumns
  private val totalRows = acc_banks * acc_bank_entries
  private val totalElements = totalRows * blockCols
  private val elementAddrBits = 1 max log2Ceil(totalElements)
  private val fragments = vpuLanes / blockCols
  private val physicalBanks = acc_banks * acc_sub_banks
  private val bankIndexBits = 1 max log2Ceil(physicalBanks)
  private val subRows = acc_bank_entries / acc_sub_banks
  // Keep the adapter on the exact same logical-row-to-sub-bank mapping as
  // Scratchpad. SubBankAddressing currently ignores the shift, but passing the
  // array row geometry here keeps the contract intact if swizzling is restored.
  private val dmaSubBankSwizzleShift = log2Ceil(meshRows * tileRows)
  private val accRowT = Vec(meshColumns, Vec(tileColumns, accType))

  require(vpuLanes == 16,
    "the fused VPU raw accumulator contract is 16 lanes")
  require(accType.getWidth == 32,
    "the fused VPU raw accumulator contract requires FP32 ACC elements")
  require(blockCols == 8 || blockCols == 16,
    "the fused VPU supports 8- or 16-element accumulator rows")
  require(vpuLanes % blockCols == 0)
  require(fragments == 1 || fragments == 2)
  require(nReadClients > 0 && nWriteClients > 0)
  require(acc_bank_entries % acc_sub_banks == 0)
  require(acc_latency >= 2)
  // A two-row DIM8 word does not require physical sub-banks. Distinct banks
  // can accept its fragments atomically; if both rows map to the same bank,
  // the serialized path below issues them in two cycles.

  val io = IO(new Bundle {
    val vpu = Flipped(new GemminiVpuMemoryIO(
      elementAddrBits, vpuLanes, accType.getWidth, tagBits,
      nReadClients, nWriteClients))
    val memory = Vec(acc_banks, Vec(acc_sub_banks,
      new ExtAccumulatorBankIO(subRows, accRowT, acc_banks)))
    val atomic = Vec(acc_banks, Vec(acc_sub_banks,
      new ExtAccumulatorAtomicClientIO))
  })

  private val flatMemory = (for {
    bank <- 0 until acc_banks
    subBank <- 0 until acc_sub_banks
  } yield io.memory(bank)(subBank)).toSeq
  private val flatAtomic = (for {
    bank <- 0 until acc_banks
    subBank <- 0 until acc_sub_banks
  } yield io.atomic(bank)(subBank)).toSeq

  private def fragmentAddress(
      address: UInt, fragmentStride: UInt, fragment: Int): UInt =
    address + fragmentStride * fragment.U

  private def globalRow(
      address: UInt, fragmentStride: UInt, fragment: Int): UInt =
    fragmentAddress(address, fragmentStride, fragment) / blockCols.U

  private def accBank(row: UInt): UInt = row / acc_bank_entries.U

  private def rowInAccBank(row: UInt): UInt = row % acc_bank_entries.U

  private def subBank(row: UInt): UInt =
    SubBankAddressing.subBankIdx(
      rowInAccBank(row), acc_sub_banks, dmaSubBankSwizzleShift)

  private def subRow(row: UInt): UInt =
    SubBankAddressing.subBankAddr(rowInAccBank(row), acc_sub_banks)

  private def flatBank(row: UInt): UInt = {
    // Only the physical-bank index is consumed below. Keeping the quotient's
    // element-address width here would replicate wide equality comparators for
    // every bank and client even though valid requests are range checked.
    val index = accBank(row) * acc_sub_banks.U + subBank(row)
    index(bankIndexBits - 1, 0)
  }

  private def fragmentActive(mask: Vec[Bool], fragment: Int): Bool = {
    val first = fragment * blockCols
    mask.slice(first, first + blockCols).reduce(_ || _)
  }

  for (port <- 0 until physicalBanks) {
    flatMemory(port).read.req.valid := false.B
    flatMemory(port).read.req.bits := DontCare
    flatMemory(port).read.resp.ready := true.B
    flatMemory(port).write.valid := false.B
    flatMemory(port).write.bits := DontCare
    flatMemory(port).write.bits.acc := false.B
    flatMemory(port).write.bits.exwrite := false.B
    flatMemory(port).grant.valid := false.B
    flatMemory(port).grant.bits := false.B
    flatAtomic(port).readAllow := false.B
    flatAtomic(port).writeAllow := false.B
  }

  private def bankMask(
      address: UInt, fragmentStride: UInt, laneMask: Vec[Bool]): UInt = {
    (0 until fragments).map { fragment =>
      val active = fragmentActive(laneMask, fragment)
      val bank = flatBank(globalRow(address, fragmentStride, fragment))
      Mux(active, UIntToOH(bank, physicalBanks), 0.U(physicalBanks.W))
    }.reduce(_ | _)
  }

  // ----------------------------------------------------------------------
  // Reads use Scratchpad's bank-local organization: every physical bank has
  // an explicit fixed-priority selection. Client index is the priority on
  // every bank, so a DIM8 request which needs two sub-banks cannot form a
  // cyclic grant. Fragments targeting distinct banks fire atomically. If a
  // segmented request maps both fragments to one physical bank, the adapter
  // issues the lower fragment first and accepts the logical request only when
  // the upper fragment has also issued.
  // ----------------------------------------------------------------------
  private val readResponseType = new GemminiVpuMemoryReadResponse(
    vpuLanes, accType.getWidth, tagBits)
  private val readQueues = Seq.fill(nReadClients) {
    Module(new Queue(readResponseType, 1, pipe = true, flow = true))
  }

  for (client <- 0 until nReadClients) {
    io.vpu.readResponse(client) <> readQueues(client).io.deq
  }
  // This is the same one-entry response admission test as AccumulatorMem.
  // It accounts for a response entering or leaving the flow-through queue in
  // this cycle, and therefore permits a replacement request on dequeue.
  private val readQueueWillBeEmpty = (0 until nReadClients).map { client =>
    (readQueues(client).io.count +& readQueues(client).io.enq.fire) -
      readQueues(client).io.deq.fire === 0.U
  }

  private val readSerialActive = RegInit(VecInit(
    Seq.fill(nReadClients)(false.B)))
  private val readSerialRequest = Reg(Vec(nReadClients,
    new GemminiVpuMemoryReadRequest(elementAddrBits, vpuLanes, tagBits)))
  private val effectiveReadRequest = Wire(Vec(nReadClients,
    new GemminiVpuMemoryReadRequest(elementAddrBits, vpuLanes, tagBits)))
  for (client <- 0 until nReadClients) {
    effectiveReadRequest(client) := Mux(readSerialActive(client),
      readSerialRequest(client), io.vpu.readRequest(client).bits)
  }

  private val rawReadMasks = (0 until nReadClients).map { client =>
    bankMask(effectiveReadRequest(client).address,
      effectiveReadRequest(client).fragmentStride,
      effectiveReadRequest(client).laneMask)
  }
  private val readFragmentActive = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    fragmentActive(effectiveReadRequest(client).laneMask, fragment)
  }
  private val readRows = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    globalRow(effectiveReadRequest(client).address,
      effectiveReadRequest(client).fragmentStride, fragment)
  }
  private val readFlatBanks = readRows.map(_.map(flatBank))
  private val readSubRows = readRows.map(_.map(subRow))
  private val readSameBank = if (fragments == 2) {
    (0 until nReadClients).map { client =>
      readFragmentActive(client)(0) && readFragmentActive(client)(1) &&
        readFlatBanks(client)(0) === readFlatBanks(client)(1)
    }
  } else {
    Seq.fill(nReadClients)(false.B)
  }
  private val readRouteFragmentActive = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    val selectedSerialFragment = Mux(readSerialActive(client),
      (fragment == 1).B, (fragment == 0).B)
    readFragmentActive(client)(fragment) &&
      (!readSameBank(client) || selectedSerialFragment)
  }
  private val readHasFragment = (0 until nReadClients).map { client =>
    readFragmentActive(client).reduce(_ || _)
  }

  private val readClientBits = 1 max log2Ceil(nReadClients)
  private val readBankArbiters = Seq.tabulate(physicalBanks) { port =>
    val arbiter = Module(new Arbiter(UInt(readClientBits.W), nReadClients))
    for (client <- 0 until nReadClients) {
      val hitsPort = (0 until fragments).map { fragment =>
        readRouteFragmentActive(client)(fragment) &&
          readFlatBanks(client)(fragment) === port.U
      }.reduce(_ || _)
      arbiter.io.in(client).valid := io.vpu.readRequest(client).valid &&
        readQueueWillBeEmpty(client) && hitsPort
      arbiter.io.in(client).bits := client.U
    }
    arbiter
  }
  private val readBankChosen = Seq.tabulate(
    physicalBanks, nReadClients) { (port, client) =>
    readBankArbiters(port).io.out.valid &&
      readBankArbiters(port).io.out.bits === client.U
  }
  private val allReadInternallyChosen = (0 until nReadClients).map { client =>
    (0 until fragments).map { fragment =>
      !readRouteFragmentActive(client)(fragment) ||
        VecInit(readBankChosen.map(_(client)))(
          readFlatBanks(client)(fragment))
    }.reduce(_ && _)
  }
  private val allReadSelected = (0 until nReadClients).map { client =>
    (0 until fragments).map { fragment =>
      !readRouteFragmentActive(client)(fragment) ||
        VecInit(flatAtomic.map(_.readSelected))(
          readFlatBanks(client)(fragment))
    }.reduce(_ && _)
  }
  private val readPhysicalFire = (0 until nReadClients).map { client =>
    io.vpu.readRequest(client).valid && readQueueWillBeEmpty(client) &&
      readHasFragment(client) && allReadInternallyChosen(client) &&
      allReadSelected(client)
  }
  private val readFirstSerialFire = (0 until nReadClients).map { client =>
    readPhysicalFire(client) && readSameBank(client) &&
      !readSerialActive(client)
  }
  for (client <- 0 until nReadClients) {
    io.vpu.readRequest(client).ready :=
      readQueueWillBeEmpty(client) && allReadInternallyChosen(client) &&
        allReadSelected(client) && readHasFragment(client) &&
        !(readSameBank(client) && !readSerialActive(client))
  }
  private val readFire = (0 until nReadClients).map { client =>
    io.vpu.readRequest(client).fire
  }

  for (port <- 0 until physicalBanks) {
    val routeHits = (for {
      client <- 0 until nReadClients
      fragment <- 0 until fragments
    } yield readBankChosen(port)(client) &&
      allReadInternallyChosen(client) &&
      readRouteFragmentActive(client)(fragment) &&
      readFlatBanks(client)(fragment) === port.U)
    val routeRows = (for {
      client <- 0 until nReadClients
      fragment <- 0 until fragments
    } yield readSubRows(client)(fragment))
    flatMemory(port).read.req.valid := routeHits.reduce(_ || _)
    flatMemory(port).read.req.bits.addr := Mux1H(routeHits, routeRows)
    flatMemory(port).read.req.bits.full := true.B
    flatAtomic(port).readAllow := (0 until nReadClients).map { client =>
      readBankChosen(port)(client) &&
        allReadInternallyChosen(client) && allReadSelected(client)
    }.reduce(_ || _)
    readBankArbiters(port).io.out.ready :=
      (0 until nReadClients).map { client =>
        readBankChosen(port)(client) && readPhysicalFire(client)
      }.reduce(_ || _)
    assert(PopCount(VecInit(routeHits)) <= 1.U,
      "a physical ACC bank received multiple VPU reads")
    when(flatMemory(port).read.req.fire) {
      assert((0 until nReadClients).map { client =>
        readBankChosen(port)(client) && readPhysicalFire(client)
      }.reduce(_ || _),
        "a VPU ACC read fragment fired without an active request")
    }
  }

  for (client <- 0 until nReadClients) {
    for (fragment <- 0 until fragments) {
      val physicalFire = VecInit(flatMemory.map(_.read.req.fire))(
        readFlatBanks(client)(fragment))
      when(readPhysicalFire(client) &&
          readRouteFragmentActive(client)(fragment)) {
        assert(physicalFire,
          "a selected VPU ACC read fragment did not physically fire")
      }
    }
  }

  private val readSerialFirstResponsePending = RegInit(VecInit(
    Seq.fill(nReadClients)(false.B)))
  private val readSerialFirstBank = Reg(Vec(nReadClients,
    UInt(bankIndexBits.W)))
  private val readSerialFirstData = Reg(Vec(nReadClients,
    Vec(blockCols, UInt(accType.getWidth.W))))
  for (client <- 0 until nReadClients) {
    readSerialFirstResponsePending(client) := readFirstSerialFire(client)
    when(readFirstSerialFire(client)) {
      readSerialActive(client) := true.B
      readSerialRequest(client) := io.vpu.readRequest(client).bits
      readSerialFirstBank(client) := readFlatBanks(client)(0)
    }.elsewhen(readFire(client) && readSerialActive(client)) {
      readSerialActive(client) := false.B
    }

    when(readSerialActive(client)) {
      assert(io.vpu.readRequest(client).valid,
        "a serialized VPU ACC read withdrew before acceptance")
      assert(io.vpu.readRequest(client).bits.asUInt ===
        readSerialRequest(client).asUInt,
        "a serialized VPU ACC read changed while stalled")
    }
  }

  private val pendingRead = RegInit(VecInit(
    Seq.fill(nReadClients)(false.B)))
  private val pendingReadAddress = Reg(Vec(nReadClients,
    UInt(elementAddrBits.W)))
  private val pendingReadFragmentStride = Reg(Vec(nReadClients,
    UInt(elementAddrBits.W)))
  private val pendingReadMask = Reg(Vec(nReadClients,
    Vec(vpuLanes, Bool())))
  private val pendingReadTag = Reg(Vec(nReadClients, UInt(tagBits.W)))
  private val pendingReadSerialized = Reg(Vec(nReadClients, Bool()))
  for (client <- 0 until nReadClients) {
    pendingRead(client) := readFire(client)
    when(readFire(client)) {
      pendingReadAddress(client) := effectiveReadRequest(client).address
      pendingReadFragmentStride(client) :=
        effectiveReadRequest(client).fragmentStride
      pendingReadMask(client) := effectiveReadRequest(client).laneMask
      pendingReadTag(client) := effectiveReadRequest(client).tag
      pendingReadSerialized(client) := readSerialActive(client)
    }
  }

  private val pendingReadRows = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    globalRow(pendingReadAddress(client),
      pendingReadFragmentStride(client), fragment)
  }
  private val pendingReadFlatBanks =
    pendingReadRows.map(_.map(flatBank))
  private val pendingReadFragmentActive = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    fragmentActive(pendingReadMask(client), fragment)
  }
  private val flatReadResponseValid =
    VecInit(flatMemory.map(_.read.resp.valid))
  private val flatReadResponseData = VecInit(flatMemory.map { memory =>
    VecInit(memory.read.resp.bits.data.flatten.map(_.asUInt))
  })
  // A client's first serialized response and its ordinary fragment-0 response
  // are mutually exclusive. Select their bank before the wide data mux so both
  // cases share one binary-indexed physical-bank selector.
  private val fragment0ResponseBank = (0 until nReadClients).map { client =>
    Mux(readSerialFirstResponsePending(client),
      readSerialFirstBank(client), pendingReadFlatBanks(client)(0))
  }
  private val fragment0ResponseValid = (0 until nReadClients).map { client =>
    flatReadResponseValid(fragment0ResponseBank(client))
  }
  private val fragment0ResponseData = (0 until nReadClients).map { client =>
    flatReadResponseData(fragment0ResponseBank(client))
  }
  for (client <- 0 until nReadClients) {
    assert(!(readSerialFirstResponsePending(client) && pendingRead(client)),
      "serialized-first and completed VPU ACC reads overlapped for one client")
    when(readSerialFirstResponsePending(client)) {
      assert(fragment0ResponseValid(client),
        "the first serialized VPU ACC read fragment did not return")
      readSerialFirstData(client) := fragment0ResponseData(client)
    }
  }
  private val pendingReadRowData = Seq.tabulate(
    nReadClients, fragments) { (client, fragment) =>
    val responseValid = if (fragment == 0) {
      fragment0ResponseValid(client)
    } else {
      flatReadResponseValid(pendingReadFlatBanks(client)(fragment))
    }
    val useSavedFirst = pendingReadSerialized(client) && (fragment == 0).B
    when(pendingRead(client) && !useSavedFirst &&
        pendingReadFragmentActive(client)(fragment)) {
      assert(responseValid,
        "an atomic VPU ACC read fragment did not return on schedule")
    }
    val physicalData = if (fragment == 0) {
      fragment0ResponseData(client)
    } else {
      flatReadResponseData(pendingReadFlatBanks(client)(fragment))
    }
    Mux(useSavedFirst, readSerialFirstData(client), physicalData)
  }

  for (client <- 0 until nReadClients) {
    val assembledData = Wire(Vec(vpuLanes,
      UInt(accType.getWidth.W)))
    for (lane <- 0 until vpuLanes) {
      val fragment = lane / blockCols
      val laneInRow = lane % blockCols
      assembledData(lane) := Mux(pendingReadMask(client)(lane),
        pendingReadRowData(client)(fragment)(laneInRow), 0.U)
    }
    readQueues(client).io.enq.valid := pendingRead(client)
    readQueues(client).io.enq.bits.data := assembledData
    readQueues(client).io.enq.bits.tag := pendingReadTag(client)
    when(pendingRead(client)) {
      assert(readQueues(client).io.enq.ready,
        "VPU ACC read response arrived while its one-entry queue was full")
    }

    when(io.vpu.readRequest(client).valid) {
      assert((io.vpu.readRequest(client).bits.address % blockCols.U) === 0.U,
        "VPU ACC read address must be aligned to one physical ACC row")
      for (fragment <- 0 until fragments) {
        when(fragmentActive(io.vpu.readRequest(client).bits.laneMask,
            fragment)) {
          val address = fragmentAddress(
            io.vpu.readRequest(client).bits.address,
            io.vpu.readRequest(client).bits.fragmentStride, fragment)
          assert((address % blockCols.U) === 0.U,
            "VPU ACC read fragment is not row aligned")
          assert(globalRow(io.vpu.readRequest(client).bits.address,
            io.vpu.readRequest(client).bits.fragmentStride,
            fragment) < totalRows.U,
            "VPU ACC read escaped accumulator capacity")
        }
      }
    }
  }

  // ----------------------------------------------------------------------
  // Writes: execute (client 0) has priority over load on overlapping bank
  // sets, while disjoint clients issue together. As on reads, a same-bank
  // fragment pair is serialized and the logical request fires on the second
  // physical write. ExtAccBank retains each accepted address until its write
  // pipeline drains and blocks a same-address read meanwhile.
  // ----------------------------------------------------------------------
  private val writeSerialActive = RegInit(VecInit(
    Seq.fill(nWriteClients)(false.B)))
  private val writeSerialRequest = Reg(Vec(nWriteClients,
    new GemminiVpuMemoryWriteRequest(
      elementAddrBits, vpuLanes, accType.getWidth)))
  private val effectiveWriteRequest = Wire(Vec(nWriteClients,
    new GemminiVpuMemoryWriteRequest(
      elementAddrBits, vpuLanes, accType.getWidth)))
  for (client <- 0 until nWriteClients) {
    effectiveWriteRequest(client) := Mux(writeSerialActive(client),
      writeSerialRequest(client), io.vpu.writeRequest(client).bits)
  }
  private val rawWriteMasks = (0 until nWriteClients).map { client =>
    bankMask(effectiveWriteRequest(client).address,
      effectiveWriteRequest(client).fragmentStride,
      effectiveWriteRequest(client).laneMask)
  }
  private val writeFragmentActive = Seq.tabulate(
    nWriteClients, fragments) { (client, fragment) =>
    fragmentActive(effectiveWriteRequest(client).laneMask, fragment)
  }
  private val writeRows = Seq.tabulate(
    nWriteClients, fragments) { (client, fragment) =>
    globalRow(effectiveWriteRequest(client).address,
      effectiveWriteRequest(client).fragmentStride, fragment)
  }
  private val writeFlatBanks = writeRows.map(_.map(flatBank))
  private val writeSubRows = writeRows.map(_.map(subRow))
  private val writeSameBank = if (fragments == 2) {
    (0 until nWriteClients).map { client =>
      writeFragmentActive(client)(0) && writeFragmentActive(client)(1) &&
        writeFlatBanks(client)(0) === writeFlatBanks(client)(1)
    }
  } else {
    Seq.fill(nWriteClients)(false.B)
  }
  private val writeRouteFragmentActive = Seq.tabulate(
    nWriteClients, fragments) { (client, fragment) =>
    val selectedSerialFragment = Mux(writeSerialActive(client),
      (fragment == 1).B, (fragment == 0).B)
    writeFragmentActive(client)(fragment) &&
      (!writeSameBank(client) || selectedSerialFragment)
  }
  private val writeHasFragment = (0 until nWriteClients).map { client =>
    writeFragmentActive(client).reduce(_ || _)
  }

  private val writeClientBits = 1 max log2Ceil(nWriteClients)
  private val writeBankArbiters = Seq.tabulate(physicalBanks) { port =>
    val arbiter = Module(new Arbiter(UInt(writeClientBits.W), nWriteClients))
    for (client <- 0 until nWriteClients) {
      val hitsPort = (0 until fragments).map { fragment =>
        writeRouteFragmentActive(client)(fragment) &&
          writeFlatBanks(client)(fragment) === port.U
      }.reduce(_ || _)
      arbiter.io.in(client).valid := io.vpu.writeRequest(client).valid &&
        hitsPort
      arbiter.io.in(client).bits := client.U
    }
    arbiter
  }
  private val writeBankChosen = Seq.tabulate(
    physicalBanks, nWriteClients) { (port, client) =>
    writeBankArbiters(port).io.out.valid &&
      writeBankArbiters(port).io.out.bits === client.U
  }
  private val allWriteInternallyChosen = (0 until nWriteClients).map { client =>
    (0 until fragments).map { fragment =>
      !writeRouteFragmentActive(client)(fragment) ||
        VecInit(writeBankChosen.map(_(client)))(
          writeFlatBanks(client)(fragment))
    }.reduce(_ && _)
  }
  private val allWriteSelected = (0 until nWriteClients).map { client =>
    (0 until fragments).map { fragment =>
      !writeRouteFragmentActive(client)(fragment) ||
        VecInit(flatAtomic.map(_.writeSelected))(
          writeFlatBanks(client)(fragment))
    }.reduce(_ && _)
  }
  private val writePhysicalFire = (0 until nWriteClients).map { client =>
    io.vpu.writeRequest(client).valid && writeHasFragment(client) &&
      allWriteInternallyChosen(client) && allWriteSelected(client)
  }
  private val writeFirstSerialFire = (0 until nWriteClients).map { client =>
    writePhysicalFire(client) && writeSameBank(client) &&
      !writeSerialActive(client)
  }
  for (client <- 0 until nWriteClients) {
    // A fully masked vector word is an architectural no-op. Accept it without
    // selecting or writing a physical ACC bank so the tagged VPU command can
    // retire instead of waiting forever for a fragment which does not exist.
    io.vpu.writeRequest(client).ready := !writeHasFragment(client) ||
      (allWriteInternallyChosen(client) && allWriteSelected(client) &&
        !(writeSameBank(client) && !writeSerialActive(client)))
  }
  private val writeFire = (0 until nWriteClients).map { client =>
    io.vpu.writeRequest(client).fire
  }

  for (port <- 0 until physicalBanks) {
    val routeHits = (for {
      client <- 0 until nWriteClients
      fragment <- 0 until fragments
    } yield writeBankChosen(port)(client) &&
      allWriteInternallyChosen(client) &&
      writeRouteFragmentActive(client)(fragment) &&
      writeFlatBanks(client)(fragment) === port.U)
    val routeRows = (for {
      client <- 0 until nWriteClients
      fragment <- 0 until fragments
    } yield writeSubRows(client)(fragment))
    val routeData = (for {
      client <- 0 until nWriteClients
      fragment <- 0 until fragments
    } yield {
      val first = fragment * blockCols
      VecInit(effectiveWriteRequest(client).data
        .slice(first, first + blockCols)).asTypeOf(accRowT)
    })
    val routeElementMasks = (for {
      client <- 0 until nWriteClients
      fragment <- 0 until fragments
    } yield {
      val first = fragment * blockCols
      VecInit(effectiveWriteRequest(client).laneMask
        .slice(first, first + blockCols))
    })
    val routeSelect = OHToUInt(VecInit(routeHits))
    val selectedRouteRow = VecInit(routeRows)(routeSelect)
    val selectedRouteData = VecInit(routeData)(routeSelect)
    val selectedElementMask = VecInit(routeElementMasks)(routeSelect)

    flatMemory(port).write.valid := routeHits.reduce(_ || _)
    flatMemory(port).write.bits.addr := selectedRouteRow
    flatMemory(port).write.bits.data := selectedRouteData
    flatMemory(port).write.bits.mask := VecInit(
      selectedElementMask.flatMap(mask =>
        Seq.fill(accType.getWidth / 8)(mask)))
    flatMemory(port).write.bits.acc := false.B
    flatMemory(port).write.bits.exwrite := false.B
    flatAtomic(port).writeAllow := (0 until nWriteClients).map { client =>
      writeBankChosen(port)(client) &&
        allWriteInternallyChosen(client) && allWriteSelected(client)
    }.reduce(_ || _)
    writeBankArbiters(port).io.out.ready :=
      (0 until nWriteClients).map { client =>
        writeBankChosen(port)(client) && writePhysicalFire(client)
      }.reduce(_ || _)
    assert(PopCount(VecInit(routeHits)) <= 1.U,
      "a physical ACC bank received multiple VPU writes")
    when(flatMemory(port).write.fire) {
      assert((0 until nWriteClients).map { client =>
        writeBankChosen(port)(client) && writePhysicalFire(client)
      }.reduce(_ || _),
        "a VPU ACC write fragment fired without an active request")
    }
  }

  for (client <- 0 until nWriteClients) {
    for (fragment <- 0 until fragments) {
      val physicalFire = VecInit(flatMemory.map(_.write.fire))(
        writeFlatBanks(client)(fragment))
      when(writePhysicalFire(client) &&
          writeRouteFragmentActive(client)(fragment)) {
        assert(physicalFire,
          "a selected VPU ACC write fragment did not physically fire")
      }
    }
  }

  for (client <- 0 until nWriteClients) {
    when(writeFirstSerialFire(client)) {
      writeSerialActive(client) := true.B
      writeSerialRequest(client) := io.vpu.writeRequest(client).bits
    }.elsewhen(writeFire(client) && writeSerialActive(client)) {
      writeSerialActive(client) := false.B
    }

    when(writeSerialActive(client)) {
      assert(io.vpu.writeRequest(client).valid,
        "a serialized VPU ACC write withdrew before acceptance")
      assert(io.vpu.writeRequest(client).bits.asUInt ===
        writeSerialRequest(client).asUInt,
        "a serialized VPU ACC write changed while stalled")
    }

    when(io.vpu.writeRequest(client).valid) {
      assert((io.vpu.writeRequest(client).bits.address % blockCols.U) === 0.U,
        "VPU ACC write address must be aligned to one physical ACC row")
      for (fragment <- 0 until fragments) {
        when(fragmentActive(io.vpu.writeRequest(client).bits.laneMask,
            fragment)) {
          val address = fragmentAddress(
            io.vpu.writeRequest(client).bits.address,
            io.vpu.writeRequest(client).bits.fragmentStride, fragment)
          assert((address % blockCols.U) === 0.U,
            "VPU ACC write fragment is not row aligned")
          assert(globalRow(io.vpu.writeRequest(client).bits.address,
            io.vpu.writeRequest(client).bits.fragmentStride,
            fragment) < totalRows.U,
            "VPU ACC write escaped accumulator capacity")
        }
      }
    }
  }

  // ReservationDeps guarantees that a VPU command never reads and writes the
  // same accumulator row in one cycle. Match AccumulatorMem's contract with
  // an assertion instead of adding another runtime dependency/scheduler here.
  for {
    readClient <- 0 until nReadClients
    writeClient <- 0 until nWriteClients
    readFragment <- 0 until fragments
    writeFragment <- 0 until fragments
  } {
    when(readFire(readClient) && writeFire(writeClient) &&
        readFragmentActive(readClient)(readFragment) &&
        writeFragmentActive(writeClient)(writeFragment)) {
      assert(readRows(readClient)(readFragment) =/=
        writeRows(writeClient)(writeFragment),
        "reading and writing the same VPU ACC row in one cycle is unsupported")
    }
  }

  private val readInternalConflict = (for {
    left <- 0 until nReadClients
    right <- left + 1 until nReadClients
  } yield io.vpu.readRequest(left).valid && readQueueWillBeEmpty(left) &&
    io.vpu.readRequest(right).valid && readQueueWillBeEmpty(right) &&
    (rawReadMasks(left) & rawReadMasks(right)).orR)
    .foldLeft(false.B)(_ || _)
  private val writeInternalConflict = (for {
    left <- 0 until nWriteClients
    right <- left + 1 until nWriteClients
  } yield io.vpu.writeRequest(left).valid &&
    io.vpu.writeRequest(right).valid &&
    (rawWriteMasks(left) & rawWriteMasks(right)).orR)
    .foldLeft(false.B)(_ || _)
  private val readRequestPending =
    VecInit(io.vpu.readRequest.map(_.valid)).asUInt.orR
  private val writeRequestPending =
    VecInit(io.vpu.writeRequest.map(_.valid)).asUInt.orR
  io.vpu.readConflictStall := readInternalConflict
  io.vpu.writeConflictStall := writeInternalConflict
  io.vpu.busy := readRequestPending || writeRequestPending ||
    readSerialActive.asUInt.orR || writeSerialActive.asUInt.orR ||
    readSerialFirstResponsePending.asUInt.orR || pendingRead.asUInt.orR ||
    readQueues.map(_.io.deq.valid).reduce(_ || _)
}
