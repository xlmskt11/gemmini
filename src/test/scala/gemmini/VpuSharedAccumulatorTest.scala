package gemmini

import chisel3._
import chisel3.iotesters.{ChiselFlatSpec, PeekPokeTester}
import chisel3.stage.ChiselStage
import freechips.rocketchip.system.DefaultConfig
import org.chipsalliance.cde.config.Parameters

import Arithmetic.FloatArithmetic._

object VpuSharedAccumulatorTestConfigs {
  private val common = GemminiFPConfigs.defaultFPConfig.copy(
    inputType = Float(8, 24),
    spatialArrayOutputType = Float(8, 24),
    accType = Float(8, 24),
    tileRows = 1,
    tileColumns = 1,
    meshRows = 8,
    meshColumns = 8,
    dataflow = Dataflow.WS,
    sp_banks = 1,
    sp_sub_banks = 4,
    sp_capacity = CapacityInMatrices(8),
    acc_banks = 1,
    acc_sub_banks = 4,
    acc_capacity = CapacityInMatrices(4),
    acc_latency = 4,
    use_shared_ext_mem = true,
    use_shared_res_entries = true,
    use_vpu_fusion = true,
    nSharers = 1,
    has_training_convs = false,
    has_max_pool = false,
    has_nonlinear_activations = false)

  val dim8: GemminiArrayConfig[Float, Float, Float] = common
  val dim8FourBanks: GemminiArrayConfig[Float, Float, Float] = common.copy(
    acc_banks = 4)
  val dim16: GemminiArrayConfig[Float, Float, Float] = common.copy(
    meshRows = 16,
    meshColumns = 16)
  val dim8NoSubBanks: GemminiArrayConfig[Float, Float, Float] = common.copy(
    acc_banks = 4,
    acc_sub_banks = 1)
}

abstract class VpuSharedAccumulatorTester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float],
    blockCols: Int,
    accBanks: Int,
    accSubBanks: Int,
    protected val accLatency: Int)
    extends PeekPokeTester(c) {

  protected val nLanes = 16
  protected val nReadClients = 3
  protected val nWriteClients = 2

  protected def memory(bank: Int, subBank: Int) =
    c.io.memory(bank)(subBank)

  protected def atomic(bank: Int, subBank: Int) =
    c.io.atomic(bank)(subBank)

  protected def setReadRequest(client: Int, address: Int,
                               activeLanes: Set[Int], tag: Int,
                               fragmentStride: Int = blockCols): Unit = {
    poke(c.io.vpu.readRequest(client).valid, true)
    poke(c.io.vpu.readRequest(client).bits.address, address)
    poke(c.io.vpu.readRequest(client).bits.fragmentStride, fragmentStride)
    poke(c.io.vpu.readRequest(client).bits.tag, tag)
    for (lane <- 0 until nLanes) {
      poke(c.io.vpu.readRequest(client).bits.laneMask(lane),
        activeLanes.contains(lane))
    }
  }

  protected def clearReadRequest(client: Int): Unit =
    poke(c.io.vpu.readRequest(client).valid, false)

  protected def setWriteRequest(client: Int, address: Int,
                                activeLanes: Set[Int], data: Seq[BigInt],
                                fragmentStride: Int = blockCols): Unit = {
    require(data.size == nLanes)
    poke(c.io.vpu.writeRequest(client).valid, true)
    poke(c.io.vpu.writeRequest(client).bits.address, address)
    poke(c.io.vpu.writeRequest(client).bits.fragmentStride, fragmentStride)
    for (lane <- 0 until nLanes) {
      poke(c.io.vpu.writeRequest(client).bits.laneMask(lane),
        activeLanes.contains(lane))
      poke(c.io.vpu.writeRequest(client).bits.data(lane), data(lane))
    }
  }

  protected def clearWriteRequest(client: Int): Unit =
    poke(c.io.vpu.writeRequest(client).valid, false)

  protected def setReadBankSelected(bank: Int, subBank: Int,
                                    selected: Boolean): Unit = {
    poke(atomic(bank, subBank).readSelected, selected)
  }

  protected def setReadBankReady(bank: Int, subBank: Int,
                                 ready: Boolean): Unit =
    poke(memory(bank, subBank).read.req.ready, ready)

  protected def setWriteBankSelected(bank: Int, subBank: Int,
                                     selected: Boolean): Unit = {
    poke(atomic(bank, subBank).writeSelected, selected)
  }

  protected def setWriteBankReady(bank: Int, subBank: Int,
                                  ready: Boolean): Unit =
    poke(memory(bank, subBank).write.ready, ready)

  protected def driveReadResponse(bank: Int, subBank: Int,
                                  data: Seq[BigInt]): Unit = {
    require(data.size == blockCols)
    val response = memory(bank, subBank).read.resp
    poke(response.valid, true)
    poke(response.bits.acc_bank_id, bank)
    response.bits.data.flatten.zip(data).foreach { case (element, value) =>
      poke(element.bits, value)
    }
  }

  protected def clearReadResponse(bank: Int, subBank: Int): Unit =
    poke(memory(bank, subBank).read.resp.valid, false)

  protected def expectReadResponse(client: Int, tag: Int,
                                   data: Seq[BigInt]): Unit = {
    require(data.size == nLanes)
    val response = c.io.vpu.readResponse(client)
    expect(response.valid, true)
    expect(response.bits.tag, tag)
    response.bits.data.zip(data).foreach { case (element, value) =>
      expect(element, value)
    }
  }

  protected def expectWriteFragment(bank: Int, subBank: Int,
                                    expectedData: Seq[BigInt],
                                    activeElements: Set[Int],
                                    expectedSubRow: Int): Unit = {
    require(expectedData.size == blockCols)
    val write = memory(bank, subBank).write
    expect(write.valid, true)
    expect(write.bits.addr, expectedSubRow)
    write.bits.data.flatten.zip(expectedData).foreach {
      case (element, value) => expect(element.bits, value)
    }
    for (element <- 0 until blockCols; byte <- 0 until 4) {
      expect(write.bits.mask(element * 4 + byte),
        activeElements.contains(element))
    }
    expect(write.bits.acc, false)
    expect(write.bits.exwrite, false)
  }

  protected def initialize(): Unit = {
    for (client <- 0 until nReadClients) {
      poke(c.io.vpu.readRequest(client).valid, false)
      poke(c.io.vpu.readRequest(client).bits.address, 0)
      poke(c.io.vpu.readRequest(client).bits.fragmentStride, blockCols)
      poke(c.io.vpu.readRequest(client).bits.tag, 0)
      c.io.vpu.readRequest(client).bits.laneMask.foreach(poke(_, false))
      poke(c.io.vpu.readResponse(client).ready, false)
    }
    for (client <- 0 until nWriteClients) {
      poke(c.io.vpu.writeRequest(client).valid, false)
      poke(c.io.vpu.writeRequest(client).bits.address, 0)
      poke(c.io.vpu.writeRequest(client).bits.fragmentStride, blockCols)
      c.io.vpu.writeRequest(client).bits.laneMask.foreach(poke(_, false))
      c.io.vpu.writeRequest(client).bits.data.foreach(poke(_, 0))
    }
    for (bank <- 0 until accBanks; subBank <- 0 until accSubBanks) {
      val port = memory(bank, subBank)
      poke(port.read.req.ready, false)
      poke(port.read.resp.valid, false)
      poke(port.read.resp.bits.acc_bank_id, 0)
      port.read.resp.bits.data.flatten.foreach(element => poke(element.bits, 0))
      poke(port.write.ready, false)
      poke(port.grant.ready, false)
      poke(atomic(bank, subBank).readSelected, false)
      poke(atomic(bank, subBank).writeSelected, false)
    }
    step(2)
  }

  protected def consumeReadResponse(client: Int): Unit = {
    poke(c.io.vpu.readResponse(client).ready, true)
    step(1)
    poke(c.io.vpu.readResponse(client).ready, false)
  }

}

class VpuSharedAccumulatorDim8Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 8,
      accBanks = 1, accSubBanks = 4, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val firstRow = (0 until 8).map(i => BigInt(0x100 + i))
  val secondRow = (0 until 8).map(i => BigInt(0x200 + i))

  // Both consecutive rows remain presented, but neither is allowed to fire
  // until both independently-arbitrated physical sub-banks select the VPU.
  setReadRequest(client = 0, address = 0, fullMask, tag = 5)
  setReadBankSelected(bank = 0, subBank = 0, selected = true)
  expect(c.io.vpu.readRequest(0).ready, false)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 1).read.req.valid, true)
  expect(atomic(0, 0).readAllow, false)
  expect(atomic(0, 1).readAllow, false)
  step(1)

  setReadBankSelected(bank = 0, subBank = 1, selected = true)
  setReadBankReady(0, 0, ready = true)
  setReadBankReady(0, 1, ready = true)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(atomic(0, 0).readAllow, true)
  expect(atomic(0, 1).readAllow, true)
  expect(memory(0, 0).read.req.bits.addr, 0)
  expect(memory(0, 1).read.req.bits.addr, 0)
  step(1)

  clearReadRequest(0)
  setReadBankSelected(0, 0, selected = false)
  setReadBankSelected(0, 1, selected = false)
  setReadBankReady(0, 0, ready = false)
  setReadBankReady(0, 1, ready = false)
  driveReadResponse(0, 0, firstRow)
  driveReadResponse(0, 1, secondRow)
  step(1)
  clearReadResponse(0, 0)
  clearReadResponse(0, 1)
  expectReadResponse(0, tag = 5, firstRow ++ secondRow)

  // Match AccumulatorMem's one-entry pipe/flow response queue. A buffered
  // response blocks another request from the same client, but consuming that
  // response allows its replacement request to fire in the same cycle.
  val replacementFirst = (0 until 8).map(i => BigInt(0x240 + i))
  val replacementSecond = (0 until 8).map(i => BigInt(0x280 + i))
  setReadRequest(client = 0, address = 16, fullMask, tag = 7)
  expect(c.io.vpu.readRequest(0).ready, false)
  expect(memory(0, 2).read.req.valid, false)
  expect(memory(0, 3).read.req.valid, false)

  poke(c.io.vpu.readResponse(0).ready, true)
  setReadBankSelected(0, 2, selected = true)
  setReadBankSelected(0, 3, selected = true)
  setReadBankReady(0, 2, ready = true)
  setReadBankReady(0, 3, ready = true)
  expectReadResponse(0, tag = 5, firstRow ++ secondRow)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(memory(0, 2).read.req.valid, true)
  expect(memory(0, 3).read.req.valid, true)
  expect(atomic(0, 2).readAllow, true)
  expect(atomic(0, 3).readAllow, true)
  step(1)

  poke(c.io.vpu.readResponse(0).ready, false)
  clearReadRequest(0)
  setReadBankSelected(0, 2, selected = false)
  setReadBankSelected(0, 3, selected = false)
  setReadBankReady(0, 2, ready = false)
  setReadBankReady(0, 3, ready = false)
  driveReadResponse(0, 2, replacementFirst)
  driveReadResponse(0, 3, replacementSecond)
  step(1)
  clearReadResponse(0, 2)
  clearReadResponse(0, 3)
  expectReadResponse(0, tag = 7, replacementFirst ++ replacementSecond)
  consumeReadResponse(0)

  // A DIM8 tail activates only one row fragment. Inactive lanes are zeroed in
  // the assembled response and the following sub-bank is never requested.
  val tailLanes = Set(0, 3, 7)
  val tailRow = (0 until 8).map(i => BigInt(0x300 + i))
  setReadRequest(client = 2, address = 32, tailLanes, tag = 6)
  setReadBankSelected(0, 0, selected = true)
  setReadBankReady(0, 0, ready = true)
  expect(c.io.vpu.readRequest(2).ready, true)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 1)
  for (subBank <- 1 until 4) {
    expect(memory(0, subBank).read.req.valid, false)
  }
  step(1)

  clearReadRequest(2)
  setReadBankSelected(0, 0, selected = false)
  setReadBankReady(0, 0, ready = false)
  driveReadResponse(0, 0, tailRow)
  step(1)
  clearReadResponse(0, 0)
  val expectedTail = (0 until nLanes).map { lane =>
    if (tailLanes.contains(lane)) tailRow(lane) else BigInt(0)
  }
  expectReadResponse(2, tag = 6, expectedTail)
  consumeReadResponse(2)

  // A full write uses the same atomic two-sub-bank admission. Like Gemmini,
  // it completes as soon as every required physical bank accepts it.
  val writeData = (0 until nLanes).map(i => BigInt(0x400 + i))
  setWriteRequest(client = 1, address = 16, fullMask, writeData)
  setWriteBankSelected(0, 2, selected = true)
  expect(c.io.vpu.writeRequest(1).ready, false)
  expect(atomic(0, 2).writeAllow, false)
  expect(atomic(0, 3).writeAllow, false)
  expectWriteFragment(0, 2, writeData.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 3, writeData.drop(8), (0 until 8).toSet,
    expectedSubRow = 0)
  step(1)

  setWriteBankSelected(0, 3, selected = true)
  setWriteBankReady(0, 2, ready = true)
  setWriteBankReady(0, 3, ready = true)
  expect(c.io.vpu.writeRequest(1).ready, true)
  expect(atomic(0, 2).writeAllow, true)
  expect(atomic(0, 3).writeAllow, true)
  step(1)
  clearWriteRequest(1)
  setWriteBankSelected(0, 2, selected = false)
  setWriteBankSelected(0, 3, selected = false)
  setWriteBankReady(0, 2, ready = false)
  setWriteBankReady(0, 3, ready = false)

  // Element masks expand to four byte enables apiece and an inactive upper
  // fragment never presents a physical write request.
  val sparseLanes = Set(0, 2, 7)
  setWriteRequest(client = 0, address = 32, sparseLanes, writeData)
  expectWriteFragment(0, 0, writeData.take(8), sparseLanes,
    expectedSubRow = 1)
  for (subBank <- 1 until 4) {
    expect(memory(0, subBank).write.valid, false)
  }
  clearWriteRequest(0)

  // A fully masked word is accepted as a no-op. No physical bank selection or
  // SRAM write is required, otherwise a legal masked VPU command deadlocks at
  // its final writeback.
  setWriteRequest(client = 0, address = 32, Set.empty, writeData)
  expect(c.io.vpu.writeRequest(0).ready, true)
  for (subBank <- 0 until 4) {
    expect(memory(0, subBank).write.valid, false)
    expect(atomic(0, subBank).writeAllow, false)
  }
  step(1)
  clearWriteRequest(0)
}

class VpuSharedAccumulatorDim16Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 16,
      accBanks = 1, accSubBanks = 4, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val rowData = (0 until nLanes).map(i => BigInt(0x500 + i))

  // DIM16 maps the complete VPU vector to one physical ACC row and requires
  // only that row's sub-bank selection.
  setReadRequest(client = 1, address = 0, fullMask, tag = 3)
  expect(c.io.vpu.readRequest(1).ready, false)
  setReadBankSelected(0, 0, selected = true)
  setReadBankReady(0, 0, ready = true)
  expect(c.io.vpu.readRequest(1).ready, true)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 0)
  for (subBank <- 1 until 4) {
    expect(memory(0, subBank).read.req.valid, false)
  }
  step(1)

  clearReadRequest(1)
  setReadBankSelected(0, 0, selected = false)
  setReadBankReady(0, 0, ready = false)
  driveReadResponse(0, 0, rowData)
  step(1)
  clearReadResponse(0, 0)
  expectReadResponse(1, tag = 3, rowData)
  consumeReadResponse(1)

  val activeLanes = Set(0, 8, 15)
  setWriteRequest(client = 0, address = 16, activeLanes, rowData)
  setWriteBankSelected(0, 1, selected = true)
  setWriteBankReady(0, 1, ready = true)
  expect(c.io.vpu.writeRequest(0).ready, true)
  expectWriteFragment(0, 1, rowData, activeLanes, expectedSubRow = 0)
  for (subBank <- Seq(0, 2, 3)) {
    expect(memory(0, subBank).write.valid, false)
  }
  step(1)
  clearWriteRequest(0)
  setWriteBankSelected(0, 1, selected = false)
  setWriteBankReady(0, 1, ready = false)
}

/** Production shared-4x8 bank geometry. Exercise binary-indexed response and
  * write routing at both the lowest and highest physical-bank numbers.
  */
class VpuSharedAccumulatorFourBankDim8Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 8,
      accBanks = 4, accSubBanks = 4, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val low0 = (0 until 8).map(i => BigInt(0x4100 + i))
  val low1 = (0 until 8).map(i => BigInt(0x4200 + i))
  val high0 = (0 until 8).map(i => BigInt(0x4300 + i))
  val high1 = (0 until 8).map(i => BigInt(0x4400 + i))

  // Client 0 selects physical banks 0/1. Client 1 selects physical banks
  // 14/15 (logical bank 3, sub-banks 2/3). Both requests issue together.
  setReadRequest(client = 0, address = 0, fullMask, tag = 1)
  setReadRequest(client = 1, address = 208, fullMask, tag = 2)
  for ((bank, subBank) <- Seq((0, 0), (0, 1), (3, 2), (3, 3))) {
    setReadBankSelected(bank, subBank, selected = true)
    setReadBankReady(bank, subBank, ready = true)
  }
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(c.io.vpu.readRequest(1).ready, true)
  expect(memory(0, 0).read.req.bits.addr, 0)
  expect(memory(0, 1).read.req.bits.addr, 0)
  expect(memory(3, 2).read.req.bits.addr, 0)
  expect(memory(3, 3).read.req.bits.addr, 0)
  step(1)

  clearReadRequest(0)
  clearReadRequest(1)
  for ((bank, subBank) <- Seq((0, 0), (0, 1), (3, 2), (3, 3))) {
    setReadBankSelected(bank, subBank, selected = false)
    setReadBankReady(bank, subBank, ready = false)
  }
  driveReadResponse(0, 0, low0)
  driveReadResponse(0, 1, low1)
  driveReadResponse(3, 2, high0)
  driveReadResponse(3, 3, high1)
  step(1)
  clearReadResponse(0, 0)
  clearReadResponse(0, 1)
  clearReadResponse(3, 2)
  clearReadResponse(3, 3)
  expectReadResponse(0, tag = 1, low0 ++ low1)
  expectReadResponse(1, tag = 2, high0 ++ high1)
  consumeReadResponse(0)
  consumeReadResponse(1)

  val lowWrite = (0 until nLanes).map(i => BigInt(0x4500 + i))
  val highWrite = (0 until nLanes).map(i => BigInt(0x4600 + i))
  setWriteRequest(client = 0, address = 16, fullMask, lowWrite)
  setWriteRequest(client = 1, address = 192, fullMask, highWrite)
  for ((bank, subBank) <- Seq((0, 2), (0, 3), (3, 0), (3, 1))) {
    setWriteBankSelected(bank, subBank, selected = true)
    setWriteBankReady(bank, subBank, ready = true)
  }
  expect(c.io.vpu.writeRequest(0).ready, true)
  expect(c.io.vpu.writeRequest(1).ready, true)
  expectWriteFragment(0, 2, lowWrite.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 3, lowWrite.drop(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(3, 0, highWrite.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(3, 1, highWrite.drop(8), (0 until 8).toSet,
    expectedSubRow = 0)
  step(1)
  clearWriteRequest(0)
  clearWriteRequest(1)
}

/** Private DIM8 geometry: four logical banks and no physical sub-banks. Two
  * adjacent row fragments in one bank must serialize without changing the
  * architectural 16-lane request.
  */
class VpuPrivateAccumulatorDim8Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 8,
      accBanks = 4, accSubBanks = 1, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val lowRead = (0 until 8).map(i => BigInt(0x5100 + i))
  val highRead = (0 until 8).map(i => BigInt(0x5200 + i))

  setReadRequest(client = 0, address = 0, fullMask, tag = 11)
  setReadBankSelected(bank = 0, subBank = 0, selected = true)
  setReadBankReady(bank = 0, subBank = 0, ready = true)
  expect(c.io.vpu.readRequest(0).ready, false)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 0)
  step(1)

  // The first bank response returns while the upper fragment issues. The
  // logical request fires only for this second physical row.
  driveReadResponse(0, 0, lowRead)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 1)
  step(1)

  clearReadRequest(0)
  clearReadResponse(0, 0)
  setReadBankSelected(0, 0, selected = false)
  setReadBankReady(0, 0, ready = false)
  driveReadResponse(0, 0, highRead)
  step(1)
  clearReadResponse(0, 0)
  expectReadResponse(0, tag = 11, lowRead ++ highRead)
  consumeReadResponse(0)

  val writeData = (0 until nLanes).map(i => BigInt(0x5300 + i))
  setWriteRequest(client = 1, address = 0, fullMask, writeData)
  setWriteBankSelected(bank = 0, subBank = 0, selected = true)
  setWriteBankReady(bank = 0, subBank = 0, ready = true)
  expect(c.io.vpu.writeRequest(1).ready, false)
  expectWriteFragment(0, 0, writeData.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  step(1)

  expect(c.io.vpu.writeRequest(1).ready, true)
  expectWriteFragment(0, 0, writeData.drop(8), (0 until 8).toSet,
    expectedSubRow = 1)
  step(1)
  clearWriteRequest(1)
}

/** Exercises non-unit DIM8 fragment strides independently from the ordinary
  * consecutive-row regressions above. Distinct physical sub-banks retain the
  * one-cycle atomic path, while a same-bank pair consumes two physical cycles
  * and fires the single logical request only on the upper fragment.
  */
class VpuSharedAccumulatorStrideDim8Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 8,
      accBanks = 1, accSubBanks = 4, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val lowRead = (0 until 8).map(i => BigInt(0x600 + i))
  val highRead = (0 until 8).map(i => BigInt(0x700 + i))

  // address 0 and address 24 select rows 0 and 3, hence sub-banks 0 and 3.
  // Both physical requests and the logical request fire together.
  setReadRequest(client = 0, address = 0, fullMask, tag = 8,
    fragmentStride = 24)
  setReadBankSelected(0, 0, selected = true)
  setReadBankSelected(0, 3, selected = true)
  setReadBankReady(0, 0, ready = true)
  setReadBankReady(0, 3, ready = true)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 0)
  expect(memory(0, 3).read.req.valid, true)
  expect(memory(0, 3).read.req.bits.addr, 0)
  expect(atomic(0, 0).readAllow, true)
  expect(atomic(0, 3).readAllow, true)
  for (subBank <- Seq(1, 2)) {
    expect(memory(0, subBank).read.req.valid, false)
  }
  step(1)

  clearReadRequest(0)
  setReadBankSelected(0, 0, selected = false)
  setReadBankSelected(0, 3, selected = false)
  setReadBankReady(0, 0, ready = false)
  setReadBankReady(0, 3, ready = false)
  driveReadResponse(0, 0, lowRead)
  driveReadResponse(0, 3, highRead)
  step(1)
  clearReadResponse(0, 0)
  clearReadResponse(0, 3)
  expectReadResponse(0, tag = 8, lowRead ++ highRead)
  consumeReadResponse(0)

  val distinctWrite = (0 until nLanes).map(i => BigInt(0x800 + i))
  // address 8 and address 32 select sub-banks 1 and 0. The write request has
  // one logical fire while both physical fragments fire in that same cycle.
  setWriteRequest(client = 0, address = 8, fullMask, distinctWrite,
    fragmentStride = 24)
  setWriteBankSelected(0, 0, selected = true)
  setWriteBankSelected(0, 1, selected = true)
  setWriteBankReady(0, 0, ready = true)
  setWriteBankReady(0, 1, ready = true)
  expect(c.io.vpu.writeRequest(0).ready, true)
  expectWriteFragment(0, 1, distinctWrite.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 0, distinctWrite.drop(8), (0 until 8).toSet,
    expectedSubRow = 1)
  for (subBank <- Seq(2, 3)) {
    expect(memory(0, subBank).write.valid, false)
  }
  step(1)
  clearWriteRequest(0)
  setWriteBankSelected(0, 0, selected = false)
  setWriteBankSelected(0, 1, selected = false)
  setWriteBankReady(0, 0, ready = false)
  setWriteBankReady(0, 1, ready = false)

  val serialLow = (0 until 8).map(i => BigInt(0x900 + i))
  val serialHigh = (0 until 8).map(i => BigInt(0xa00 + i))
  // address 32 and address 96 select rows 4 and 12. Both map to sub-bank 0,
  // so the lower physical read fires first while logical ready stays low.
  setReadRequest(client = 1, address = 32, fullMask, tag = 9,
    fragmentStride = 64)
  setReadBankSelected(0, 0, selected = true)
  setReadBankReady(0, 0, ready = true)
  expect(c.io.vpu.readRequest(1).ready, false)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 1)
  expect(atomic(0, 0).readAllow, true)
  for (subBank <- 1 until 4) {
    expect(memory(0, subBank).read.req.valid, false)
  }
  step(1)

  // The first response returns on schedule, but hold the second fragment for
  // several cycles to exercise the retained serialized request state.
  setReadBankSelected(0, 0, selected = false)
  setReadBankReady(0, 0, ready = false)
  driveReadResponse(0, 0, serialLow)
  expect(c.io.vpu.readRequest(1).ready, false)
  expect(memory(0, 0).read.req.valid, true)
  expect(memory(0, 0).read.req.bits.addr, 3)
  step(1)
  clearReadResponse(0, 0)
  for (_ <- 0 until 2) {
    expect(c.io.vpu.readRequest(1).ready, false)
    expect(memory(0, 0).read.req.valid, true)
    expect(memory(0, 0).read.req.bits.addr, 3)
    step(1)
  }

  setReadBankSelected(0, 0, selected = true)
  setReadBankReady(0, 0, ready = true)
  expect(c.io.vpu.readRequest(1).ready, true)
  expect(memory(0, 0).read.req.bits.addr, 3)
  step(1)

  clearReadRequest(1)
  setReadBankSelected(0, 0, selected = false)
  setReadBankReady(0, 0, ready = false)
  driveReadResponse(0, 0, serialHigh)
  step(1)
  clearReadResponse(0, 0)
  expectReadResponse(1, tag = 9, serialLow ++ serialHigh)
  consumeReadResponse(1)

  val serialWrite = (0 until nLanes).map(i => BigInt(0xb00 + i))
  // Rows 5 and 13 both map to sub-bank 1. The first cycle writes only the
  // lower lanes and must not fire the logical request.
  setWriteRequest(client = 1, address = 40, fullMask, serialWrite,
    fragmentStride = 64)
  setWriteBankSelected(0, 1, selected = true)
  setWriteBankReady(0, 1, ready = true)
  expect(c.io.vpu.writeRequest(1).ready, false)
  expectWriteFragment(0, 1, serialWrite.take(8), (0 until 8).toSet,
    expectedSubRow = 1)
  for (subBank <- Seq(0, 2, 3)) {
    expect(memory(0, subBank).write.valid, false)
  }
  step(1)

  // Likewise, stall the second write fragment for several cycles.
  setWriteBankSelected(0, 1, selected = false)
  setWriteBankReady(0, 1, ready = false)
  for (_ <- 0 until 2) {
    expect(c.io.vpu.writeRequest(1).ready, false)
    expectWriteFragment(0, 1, serialWrite.drop(8), (0 until 8).toSet,
      expectedSubRow = 3)
    step(1)
  }

  setWriteBankSelected(0, 1, selected = true)
  setWriteBankReady(0, 1, ready = true)
  expect(c.io.vpu.writeRequest(1).ready, true)
  expectWriteFragment(0, 1, serialWrite.drop(8), (0 until 8).toSet,
    expectedSubRow = 3)
  for (subBank <- Seq(0, 2, 3)) {
    expect(memory(0, subBank).write.valid, false)
  }
  step(1)
  clearWriteRequest(1)
}

class VpuSharedAccumulatorParallelDim8Tester(
    c: VpuSharedAccumulatorAdapter[Float, Float, Float])
    extends VpuSharedAccumulatorTester(c, blockCols = 8,
      accBanks = 1, accSubBanks = 4, accLatency = 4) {

  initialize()

  val fullMask = (0 until nLanes).toSet
  val zeroData = Seq.fill(nLanes)(BigInt(0))

  // c0 needs sub-banks {0,1}, c1 needs {1,2}, and c2 needs {2,3}.
  // Every bank has the same fixed client priority. Bank 2 initially chooses
  // c1, so c2 is not presented externally even though c1 itself loses bank 1
  // to c0. The requests therefore make progress in c0, c1, c2 order.
  setReadRequest(client = 0, address = 0, fullMask, tag = 1)
  setReadRequest(client = 1, address = 8, fullMask, tag = 2)
  setReadRequest(client = 2, address = 16, fullMask, tag = 3)
  setReadBankSelected(0, 0, selected = true)
  setReadBankSelected(0, 1, selected = true)
  setReadBankReady(0, 0, ready = true)
  setReadBankReady(0, 1, ready = true)
  expect(c.io.vpu.readRequest(0).ready, true)
  expect(c.io.vpu.readRequest(1).ready, false)
  expect(c.io.vpu.readRequest(2).ready, false)
  expect(c.io.vpu.readConflictStall, true)
  for (subBank <- 0 until 2) {
    expect(memory(0, subBank).read.req.valid, true)
    expect(atomic(0, subBank).readAllow, true)
  }
  for (subBank <- 2 until 4) {
    expect(memory(0, subBank).read.req.valid, false)
    expect(atomic(0, subBank).readAllow, false)
  }
  step(1)

  clearReadRequest(0)
  val c0Low = (0 until 8).map(i => BigInt(0x1000 + i))
  val c0High = (0 until 8).map(i => BigInt(0x1100 + i))
  val c2Low = (0 until 8).map(i => BigInt(0x1200 + i))
  val c2High = (0 until 8).map(i => BigInt(0x1300 + i))
  driveReadResponse(0, 0, c0Low)
  driveReadResponse(0, 1, c0High)
  setReadBankSelected(0, 0, selected = false)
  setReadBankSelected(0, 2, selected = true)
  setReadBankReady(0, 0, ready = false)
  setReadBankReady(0, 2, ready = true)

  // Once c0 fires, c1 wins both of its banks. The partially-winning c2
  // fragment remains suppressed, so ExtAccBank cannot select a request which
  // the adapter would subsequently refuse with atomic allow.
  expect(c.io.vpu.readRequest(1).ready, true)
  expect(c.io.vpu.readRequest(2).ready, false)
  expect(c.io.vpu.readConflictStall, true)
  expect(memory(0, 0).read.req.valid, false)
  expect(memory(0, 1).read.req.valid, true)
  expect(memory(0, 2).read.req.valid, true)
  expect(memory(0, 3).read.req.valid, false)
  expect(atomic(0, 0).readAllow, false)
  expect(atomic(0, 1).readAllow, true)
  expect(atomic(0, 2).readAllow, true)
  expect(atomic(0, 3).readAllow, false)
  step(1)

  clearReadRequest(1)
  clearReadResponse(0, 0)
  clearReadResponse(0, 1)
  val c1Low = (0 until 8).map(i => BigInt(0x1400 + i))
  val c1High = (0 until 8).map(i => BigInt(0x1500 + i))
  driveReadResponse(0, 1, c1Low)
  driveReadResponse(0, 2, c1High)
  setReadBankSelected(0, 1, selected = false)
  setReadBankSelected(0, 3, selected = true)
  setReadBankReady(0, 1, ready = false)
  setReadBankReady(0, 3, ready = true)

  expect(c.io.vpu.readRequest(2).ready, true)
  expect(c.io.vpu.readConflictStall, false)
  expect(memory(0, 0).read.req.valid, false)
  expect(memory(0, 1).read.req.valid, false)
  expect(memory(0, 2).read.req.valid, true)
  expect(memory(0, 3).read.req.valid, true)
  expect(atomic(0, 0).readAllow, false)
  expect(atomic(0, 1).readAllow, false)
  expect(atomic(0, 2).readAllow, true)
  expect(atomic(0, 3).readAllow, true)
  step(1)

  clearReadRequest(2)
  clearReadResponse(0, 1)
  clearReadResponse(0, 2)
  for (subBank <- 0 until 4) {
    setReadBankSelected(0, subBank, selected = false)
    setReadBankReady(0, subBank, ready = false)
  }
  driveReadResponse(0, 2, c2Low)
  driveReadResponse(0, 3, c2High)
  step(1)
  clearReadResponse(0, 2)
  clearReadResponse(0, 3)

  expectReadResponse(0, tag = 1, c0Low ++ c0High)
  expectReadResponse(1, tag = 2, c1Low ++ c1High)
  expectReadResponse(2, tag = 3, c2Low ++ c2High)
  consumeReadResponse(0)
  consumeReadResponse(1)
  consumeReadResponse(2)

  // Execute and load writes to disjoint sub-bank pairs issue and complete
  // together. This is the path the former global write arbiter serialized.
  val executeData = (0 until nLanes).map(i => BigInt(0x2000 + i))
  val loadData = (0 until nLanes).map(i => BigInt(0x3000 + i))
  setWriteRequest(client = 0, address = 0, fullMask, executeData)
  setWriteRequest(client = 1, address = 16, fullMask, loadData)
  for (subBank <- 0 until 4) {
    setWriteBankSelected(0, subBank, selected = true)
    setWriteBankReady(0, subBank, ready = true)
  }
  expect(c.io.vpu.writeRequest(0).ready, true)
  expect(c.io.vpu.writeRequest(1).ready, true)
  expect(c.io.vpu.writeConflictStall, false)
  expectWriteFragment(0, 0, executeData.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 1, executeData.drop(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 2, loadData.take(8), (0 until 8).toSet,
    expectedSubRow = 0)
  expectWriteFragment(0, 3, loadData.drop(8), (0 until 8).toSet,
    expectedSubRow = 0)
  step(1)
  clearWriteRequest(0)
  clearWriteRequest(1)
  for (subBank <- 0 until 4) {
    setWriteBankSelected(0, subBank, selected = false)
    setWriteBankReady(0, subBank, ready = false)
  }

  // On an overlapping pair, client 0 is the execute writer and therefore
  // owns the bank set ahead of client 1 (load), matching Gemmini's local
  // execute-before-load selection. The VPU request remains non-exwrite at the
  // outer ExtAccBank boundary.
  setWriteRequest(client = 0, address = 8, fullMask, zeroData)
  setWriteRequest(client = 1, address = 16, fullMask, zeroData)
  setWriteBankSelected(0, 1, selected = true)
  setWriteBankSelected(0, 2, selected = true)
  setWriteBankReady(0, 1, ready = true)
  setWriteBankReady(0, 2, ready = true)
  expect(c.io.vpu.writeRequest(0).ready, true)
  expect(c.io.vpu.writeRequest(1).ready, false)
  expect(c.io.vpu.writeConflictStall, true)
  expect(memory(0, 1).write.valid, true)
  expect(memory(0, 2).write.valid, true)
  expect(memory(0, 3).write.valid, false)
  expect(atomic(0, 1).writeAllow, true)
  expect(atomic(0, 2).writeAllow, true)
  expect(atomic(0, 3).writeAllow, false)
  step(1)

  clearWriteRequest(0)
  setWriteBankSelected(0, 1, selected = false)
  setWriteBankSelected(0, 3, selected = true)
  setWriteBankReady(0, 1, ready = false)
  setWriteBankReady(0, 3, ready = true)
  expect(c.io.vpu.writeRequest(1).ready, true)
  expect(c.io.vpu.writeConflictStall, false)
  expect(memory(0, 1).write.valid, false)
  expect(memory(0, 2).write.valid, true)
  expect(memory(0, 3).write.valid, true)
  expect(atomic(0, 1).writeAllow, false)
  expect(atomic(0, 2).writeAllow, true)
  expect(atomic(0, 3).writeAllow, true)
  step(1)
  clearWriteRequest(1)
}

class ExtAccBankPendingWriteTester(c: ExtAccBank[Float], accLatency: Int)
    extends PeekPokeTester(c) {
  val port = c.io.in(0)
  val atomic = c.io.atomic.get

  poke(port.read.req.valid, false)
  poke(port.read.req.bits.addr, 0)
  poke(port.read.req.bits.full, true)
  poke(port.read.resp.ready, true)
  poke(port.write.valid, false)
  poke(port.write.bits.addr, 0)
  poke(port.write.bits.acc, false)
  poke(port.write.bits.exwrite, false)
  port.write.bits.data.flatten.foreach(element => poke(element.bits, 0))
  port.write.bits.mask.foreach(poke(_, true))
  poke(port.grant.valid, false)
  poke(port.grant.bits, false)
  poke(atomic.readAllow, true)
  poke(atomic.writeAllow, true)
  c.io.adder.sum.flatten.foreach(element => poke(element.bits, 0))
  step(2)

  // Completion occurs on this acceptance edge, while ExtAccBank retains the
  // address in pipelined_writes until the physical SRAM update.
  poke(port.write.valid, true)
  expect(port.write.ready, true)
  expect(atomic.writeSelected, true)
  step(1)
  poke(port.write.valid, false)

  // A different row remains readable; only the pending write address stalls.
  poke(port.read.req.valid, true)
  poke(port.read.req.bits.addr, 1)
  expect(atomic.readSelected, true)
  expect(port.read.req.ready, true)
  poke(port.read.req.bits.addr, 0)
  for (_ <- 0 until accLatency) {
    expect(atomic.readSelected, false)
    expect(port.read.req.ready, false)
    step(1)
  }
  expect(atomic.readSelected, true)
  expect(port.read.req.ready, true)
}

class VpuSharedAccumulatorUnitTest extends ChiselFlatSpec {
  behavior of "VpuSharedAccumulatorAdapter"

  it should "atomically gather DIM8 rows and complete writes on acceptance" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-dim8")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim8, vpuLanes = 16,
        tagBits = 4)) { c =>
      new VpuSharedAccumulatorDim8Tester(c)
    } should be(true)
  }

  it should "map a DIM16 VPU request to one physical accumulator row" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-dim16")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim16, vpuLanes = 16,
        tagBits = 4)) { c =>
      new VpuSharedAccumulatorDim16Tester(c)
    } should be(true)
  }

  it should "route the lowest and highest shared4x8 physical banks" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-four-bank-dim8")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim8FourBanks,
        vpuLanes = 16, tagBits = 4)) { c =>
      new VpuSharedAccumulatorFourBankDim8Tester(c)
    } should be(true)
  }

  it should "serialize a DIM8 word across one bank without sub-banks" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-private-acc-dim8-no-subbanks")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim8NoSubBanks,
        vpuLanes = 16, tagBits = 4)) { c =>
      new VpuPrivateAccumulatorDim8Tester(c)
    } should be(true)
  }

  it should "route strided DIM8 fragments in parallel or serially by bank" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-stride-dim8")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim8, vpuLanes = 16,
        tagBits = 4)) { c =>
      new VpuSharedAccumulatorStrideDim8Tester(c)
    } should be(true)
  }

  it should "apply per-bank DIM8 priority and accept disjoint writes concurrently" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-parallel-dim8")
    chisel3.iotesters.Driver.execute(args, () =>
      new VpuSharedAccumulatorAdapter(
        VpuSharedAccumulatorTestConfigs.dim8, vpuLanes = 16,
        tagBits = 4)) { c =>
      new VpuSharedAccumulatorParallelDim8Tester(c)
    } should be(true)
  }

  it should "block a same-address read while an accepted ACC write is pending" in {
    val args = Array("--backend-name", "treadle",
      "--target-dir", "test_run_dir/vpu-shared-acc-pending-write")
    chisel3.iotesters.Driver.execute(args, () =>
      new ExtAccBank[Float](
        nSharers = 1,
        n = 16,
        t = Vec(8, Vec(1, Float(8, 24))),
        accBanks = 1,
        acc_singleported = false,
        acc_latency = 4,
        enableExWrite = true,
        atomicClient = 0)) { c =>
      new ExtAccBankPendingWriteTester(c, accLatency = 4)
    } should be(true)
  }

  it should "elaborate SharedExtMem_4 for both fused array geometries" in {
    implicit val p: Parameters = (new DefaultConfig).toInstance
    val dim8 = (new ChiselStage).emitChirrtl(new SharedExtMem_4(
      VpuSharedAccumulatorTestConfigs.dim8, vpuLanes = 16,
      vpuTagBits = 4))
    val dim16 = (new ChiselStage).emitChirrtl(new SharedExtMem_4(
      VpuSharedAccumulatorTestConfigs.dim16, vpuLanes = 16,
      vpuTagBits = 4))
    dim8 should include("module SharedExtMem_4")
    dim16 should include("module SharedExtMem_4")
  }
}
