package gemmini

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import Util._

class ExtMemIO extends Bundle {
  val read_en = Output(Bool())
  val read_addr = Output(UInt())
  val read_data = Input(UInt())

  val write_en = Output(Bool())
  val write_addr = Output(UInt())
  val write_data = Output(UInt())
  val write_mask = Output(UInt())
}

// made
class ExtMemIO_4 extends Bundle {
  val read_en = Output(Bool())
  val read_addr = Output(UInt())
  val read_data = Input(UInt())
  // val read_valid = Input(Bool())

  val write_en = Output(Bool())
  val write_addr = Output(UInt())
  val write_data = Output(UInt())
  val write_mask = Output(UInt())
}

class ExtSpadMemIO(sp_banks: Int, acc_banks: Int, acc_sub_banks: Int) extends Bundle {
  val spad = Vec(sp_banks, new ExtMemIO)
  val acc = Vec(acc_banks, Vec(acc_sub_banks, new ExtMemIO))
}

// made
class ExtSpadMemIO_4(sp_banks: Int, acc_banks: Int, acc_sub_banks: Int) extends Bundle {
  val spad = Vec(sp_banks, new ExtMemIO_4)
  val acc = Vec(acc_banks, Vec(acc_sub_banks, new ExtMemIO_4))
}

// single ported multi-port like sram start
class ExtScratchpadReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
}

class ExtScratchpadReadResp(val w: Int) extends Bundle {
  val data = UInt(w.W)
}

class ExtScratchpadReadIO(val n: Int, val w: Int) extends Bundle {
  val req = Decoupled(new ExtScratchpadReadReq(n))
  val resp = Flipped(Decoupled(new ExtScratchpadReadResp(w)))
}

class ExtScratchpadWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr = Output(UInt(log2Ceil(n).W))
  val mask = Output(Vec(mask_len, Bool()))
  val data = Output(UInt(w.W))
  val exwrite = Output(Bool())
}

class BankExWriteGrantReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
}

class ExtScratchpadBankIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val read = new ExtScratchpadReadIO(n, w)
  val write = Decoupled(new ExtScratchpadWriteReq(n, w, mask_len))
  val grant = Decoupled(Bool())
}

class ExtAccumulatorReadReq(n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val full = Bool() // Whether or not we return the full bitwidth output
}

class ExtAccumulatorReadResp[T <: Data: Arithmetic](
    fullDataType: Vec[Vec[T]], accBanks: Int) extends Bundle {
  val data = fullDataType.cloneType
  val acc_bank_id = UInt((1 max log2Ceil(accBanks)).W)
}

class ExtAccumulatorReadIO[T <: Data: Arithmetic](
    n: Int, fullDataType: Vec[Vec[T]], accBanks: Int) extends Bundle {
  val req = Decoupled(new ExtAccumulatorReadReq(n))
  val resp = Flipped(Decoupled(new ExtAccumulatorReadResp[T](fullDataType, accBanks)))
}

class ExtAccumulatorWriteReq[T <: Data: Arithmetic](
    n: Int,
    t: Vec[Vec[T]]) extends Bundle {
  val addr = UInt(log2Up(n).W)
  val data = t.cloneType
  val acc = Bool()
  val mask = Vec(t.getWidth / 8, Bool()) // TODO Use aligned_to here
  val exwrite = Bool()
}

class ExtAccumulatorBankIO[T <: Data: Arithmetic](
    n: Int,
    t: Vec[Vec[T]], accBanks: Int) extends Bundle {
  val read = new ExtAccumulatorReadIO(n, t, accBanks)
  val write = Decoupled(new ExtAccumulatorWriteReq(
    n, t))
  val grant = Decoupled(Bool())
}

/** Sideband used by the unified VPU client to make a multi-row request
  * indivisible. A physical bank reports that it selected the VPU before its
  * Decoupled ready is exposed; the adapter enables every selected fragment
  * only when all required banks selected the same logical request. */
class ExtAccumulatorAtomicClientIO extends Bundle {
  val readSelected = Input(Bool())
  val writeSelected = Input(Bool())
  val readAllow = Output(Bool())
  val writeAllow = Output(Bool())
}

class ExtMemIO_new[T <: Data: Arithmetic](
    sp_banks: Int, sp_sub_banks: Int, sp_n: Int, sp_w: Int,
    sp_mask_len: Int, acc_banks: Int, acc_sub_banks: Int, acc_n: Int,
    acc_t: Vec[Vec[T]])
    extends Bundle {
  val spad = Vec(sp_banks, Vec(sp_sub_banks, new ExtScratchpadBankIO(sp_n, sp_w, sp_mask_len)))
  val acc = Vec(acc_banks, Vec(acc_sub_banks, new ExtAccumulatorBankIO(
    acc_n, acc_t, acc_banks)))
}

class AdderSellector(nSharers: Int, banks: Int) extends Module {
  val io = IO(new Bundle {
    val in_valid = Input(Vec(banks, Bool()))
    val selected_oh = Output(Vec(nSharers, UInt(banks.W)))
    // val selected_index = Output(Vec(nSharers, Valid(UInt(log2Ceil(banks).W))))
  })

  assert(PopCount(io.in_valid) <= nSharers.U, "you cannot select more than (nSharers) adders input at a time")
  val valid_list = Wire(Vec(nSharers + 1, UInt(banks.W)))
  valid_list(0) := io.in_valid.asUInt
  for (i <- 0 until nSharers) {
    val pickOH = PriorityEncoderOH(valid_list(i))
    val pickedid = OHToUInt(pickOH)
    io.selected_oh(i) := pickOH
    // io.selected_index(i).valid := pickOH.orR
    // io.selected_index(i).bits := pickedid
    valid_list(i + 1) := valid_list(i) & ~pickOH
  }
}


// class ExtScratchpadBank(
//   nSharers: Int, n: Int, w: Int, aligned_to: Int, single_ported: Boolean,
//   buffer_capacity: Int, max_in_flight_sram: Int, enableExWrite: Boolean = true
// ) extends Module {
//   // This is essentially a pipelined SRAM with the ability to stall pipeline stages

//   require(w % aligned_to == 0 || w < aligned_to)
//   val mask_len = (w / (aligned_to * 8)) max 1 // How many mask bits are there?
//   val mask_elem = UInt((w min (aligned_to * 8)).W) // What datatype does each mask bit correspond to?

//   val io = IO(new Bundle {
//     val in = Vec(nSharers, Flipped(new ExtScratchpadBankIO(n, w, mask_len, buffer_capacity, max_in_flight_sram)))
//   })

//   // Writes take precedence within each selected sharer

//   class ExtScratchpadReqWTag(val nSharers: Int, val n: Int, val w: Int, val mask_len: Int) extends Bundle {
//     val ren = Bool()
//     val raddr = UInt(log2Ceil(n).W)
//     val wen = Bool()
//     val waddr = UInt(log2Ceil(n).W)
//     val wdata = UInt(w.W)
//     val wmask = Vec(mask_len, Bool())
//     val fromDMA = Bool()
//     val tag = UInt(log2Ceil(nSharers).W)
//     val idx = UInt(log2Ceil(max_in_flight_sram).W)
//   }
  
//   val circbuffer = Module(new CircularBuffer(new ExtScratchpadReqWTag(nSharers, n, w, mask_len), nSharers, buffer_capacity))
//   val request_accepted = io.in.map{ case wrp => wrp.write.fire || wrp.read.req.fire}
//   val request_accepted_vec = VecInit(request_accepted)
//   val reqRrPtr = RegInit(0.U((1 max log2Ceil(nSharers)).W))

//   circbuffer.io.enqValid := PopCount(request_accepted)
//   circbuffer.io.flush := false.B
//   io.in.foreach(_.nSpace := circbuffer.io.nSpace)

//   private val reqCountBits = log2Ceil(nSharers + 1)
//   private val comingWindow = buffer_capacity
//   private val comingHBits = 1 max log2Ceil((comingWindow max 1) + 1)
//   private val defaultComingH = (comingWindow - 1) max 0

//   val grantedCount = Wire(UInt(reqCountBits.W))
//   val remindedCount = Wire(UInt(reqCountBits.W))
//   val nearestComingH = Wire(UInt(comingHBits.W))
//   nearestComingH := defaultComingH.U

//   if (enableExWrite) {
//     val gSpace = RegInit(buffer_capacity.U(log2Ceil(buffer_capacity + 1).W))
//     val comingRegs = RegInit(VecInit(Seq.fill(comingWindow)(0.U(reqCountBits.W))))
//     val comingPtr = RegInit(0.U((1 max log2Ceil(comingWindow)).W))
//     val grantRrPtr = RegInit(0.U((1 max log2Ceil(nSharers)).W))

//     val availableGrant = Mux(gSpace >= nSharers.U, nSharers.U, gSpace)

//     val grantReqMask = VecInit(io.in.map(_.grant.valid)).asUInt
//     val grantReqMaskRR = grantReqMask.rotateRight(grantRrPtr)
//     val grantCandRemaining = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//     val grantSelectedMask = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//     val grantSelectedCount = Wire(Vec(nSharers + 1, UInt(reqCountBits.W)))
//     grantCandRemaining(0) := grantReqMaskRR
//     grantSelectedMask(0) := 0.U
//     grantSelectedCount(0) := 0.U
//     for (i <- 0 until nSharers) {
//       val pickOH = PriorityEncoderOH(grantCandRemaining(i))
//       val canGrant = pickOH.orR && (grantSelectedCount(i) < availableGrant)
//       val grantOH = Mux(canGrant, pickOH, 0.U(nSharers.W))
//       grantSelectedMask(i + 1) := grantSelectedMask(i) | grantOH
//       grantCandRemaining(i + 1) := grantCandRemaining(i) & ~grantOH
//       grantSelectedCount(i + 1) := grantSelectedCount(i) + canGrant
//     }
//     val grantSelectedMaskRR = grantSelectedMask(nSharers)
//     val grantSelectedMaskOrig = grantSelectedMaskRR.rotateLeft(grantRrPtr)
//     val grantSelected = grantSelectedMaskOrig.asBools

//     for (i <- 0 until nSharers) {
//       io.in(i).grant.ready := grantSelected(i)
//       io.in(i).remind.ready := true.B
//     }

//     grantedCount := PopCount(VecInit(io.in.map(_.grant.fire)))
//     remindedCount := PopCount(VecInit(io.in.map(_.remind.fire)))

//     val rrAdvance = Mux(grantedCount === nSharers.U, 0.U, grantedCount)

//     when (grantedCount =/= 0.U) {
//       grantRrPtr := wrappingAdd(grantRrPtr, rrAdvance, nSharers)
//     }

//     if (comingWindow > 1) {
//       val hFound = Wire(Bool())
//       hFound := false.B
//       nearestComingH := defaultComingH.U
//       for (h <- 1 until comingWindow) {
//         val idx = wrappingAdd(comingPtr, h.U, comingWindow)
//         when (!hFound && comingRegs(idx) =/= 0.U) {
//           nearestComingH := h.U
//           hFound := true.B
//         }
//       }
//     }

//     when (gSpace === buffer_capacity.U) {
//       gSpace := gSpace - grantedCount
//     }.otherwise {
//       gSpace := gSpace - grantedCount + 1.U
//     }
//     comingRegs(comingPtr) := remindedCount
//     comingPtr := wrappingAdd(comingPtr, 1.U, comingWindow)

//     when (reset.asBool) {
//       gSpace := buffer_capacity.U
//     }
//   } else {
//     for (i <- 0 until nSharers) {
//       io.in(i).grant.ready := true.B
//       io.in(i).remind.ready := true.B
//     }
//     grantedCount := 0.U
//     remindedCount := 0.U
//     nearestComingH := defaultComingH.U
//   }

//   val admitRawSigned = circbuffer.io.nSpace.zext + nearestComingH.zext + 1.S - buffer_capacity.S
//   val admitRaw = Mux(admitRawSigned > 0.S, admitRawSigned.asUInt, 0.U)
//   val availableAdmit = Mux(admitRaw >= nSharers.U, nSharers.U, admitRaw)

//   val exwriteCandidate = VecInit(io.in.map(in => in.write.valid && in.write.bits.exwrite))
//   assert(PopCount(exwriteCandidate) <= 1.U)
//   val hasExwrite = exwriteCandidate.asUInt.orR

//   val nonExWriteCandidate = VecInit(io.in.map(in => in.write.valid && !in.write.bits.exwrite && !hasExwrite))
//   val readCandidate = VecInit(io.in.map(in => in.read.req.valid && !hasExwrite && !(in.write.valid && single_ported.B)))
//   val nonExCandidateMask = VecInit(nonExWriteCandidate.zip(readCandidate).map { case (wC, rC) => wC || rC }).asUInt
//   val nonExCandidateMaskRR = nonExCandidateMask.rotateRight(reqRrPtr)

//   val nonExCandRemaining = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   val nonExSelectedMask = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   val nonExSelectedCount = Wire(Vec(nSharers + 1, UInt(reqCountBits.W)))
//   nonExCandRemaining(0) := nonExCandidateMaskRR
//   nonExSelectedMask(0) := 0.U
//   nonExSelectedCount(0) := 0.U
//   for (i <- 0 until nSharers) {
//     val pickOH = PriorityEncoderOH(nonExCandRemaining(i))
//     val canGrant = pickOH.orR && (nonExSelectedCount(i) < availableAdmit)
//     val grantOH = Mux(canGrant, pickOH, 0.U(nSharers.W))
//     nonExSelectedMask(i + 1) := nonExSelectedMask(i) | grantOH
//     nonExCandRemaining(i + 1) := nonExCandRemaining(i) & ~grantOH
//     nonExSelectedCount(i + 1) := nonExSelectedCount(i) + canGrant
//   }
//   val nonExSelectedMaskRR = nonExSelectedMask(nSharers)
//   val nonExSelectedMaskOrig = nonExSelectedMaskRR.rotateLeft(reqRrPtr)
//   val nonExSelected = nonExSelectedMaskOrig.asBools
//   val nonExSelectedTotal = nonExSelectedCount(nSharers)

//   for (i <- 0 until nSharers) {
//     val canTakeNonExWrite = nonExSelected(i) && nonExWriteCandidate(i)
//     val canTakeRead = nonExSelected(i) && readCandidate(i)
//     io.in(i).write.ready := exwriteCandidate(i) || canTakeNonExWrite
//     io.in(i).read.req.ready := canTakeRead
//   }

//   when (!hasExwrite && nonExSelectedTotal =/= 0.U) {
//     val rrAdvance = Mux(nonExSelectedTotal === nSharers.U, 0.U, nonExSelectedTotal)
//     reqRrPtr := wrappingAdd(reqRrPtr, rrAdvance, nSharers)
//   }
//   val enqCandidates = Wire(Vec(nSharers, new ExtScratchpadReqWTag(nSharers, n, w, mask_len)))
//   for (i <- 0 until nSharers) {
//     enqCandidates(i).ren := io.in(i).read.req.fire
//     enqCandidates(i).raddr := io.in(i).read.req.bits.addr
//     enqCandidates(i).wen := io.in(i).write.fire
//     enqCandidates(i).waddr := io.in(i).write.bits.addr
//     enqCandidates(i).wdata := io.in(i).write.bits.data
//     enqCandidates(i).wmask := io.in(i).write.bits.mask
//     enqCandidates(i).fromDMA := io.in(i).read.req.bits.fromDMA
//     enqCandidates(i).idx := io.in(i).read.req.bits.idx
//     enqCandidates(i).tag := i.U
//   }

//   val remainingAccepted = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   remainingAccepted(0) := request_accepted_vec.asUInt
//   for (i <- 0 until nSharers) {
//     val pickOH = PriorityEncoderOH(remainingAccepted(i))
//     val pickedReq = Mux1H(pickOH.asBools.zip(enqCandidates))
//     circbuffer.io.enqData(i) := Mux(pickOH.orR, pickedReq, 0.U.asTypeOf(enqCandidates(i)))
//     remainingAccepted(i + 1) := remainingAccepted(i) & ~pickOH
//   }

//   val mem = SyncReadMem(n, Vec(mask_len, UInt(mask_elem.getWidth.W)))

//   when (circbuffer.io.deqFire() && circbuffer.io.dataOut.wen) {
//     if (aligned_to >= w)
//       mem.write(circbuffer.io.dataOut.waddr, circbuffer.io.dataOut.wdata.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
//     else
//       mem.write(circbuffer.io.dataOut.waddr, circbuffer.io.dataOut.wdata.asTypeOf(Vec(mask_len, mask_elem)), circbuffer.io.dataOut.wmask)
//   }

//   val raddr = circbuffer.io.dataOut.raddr
//   val ren = circbuffer.io.dataOut.ren && circbuffer.io.deqFire()
//   // changed
//   val rdata = if (single_ported) {
//     assert(!(ren && circbuffer.io.dataOut.wen))
//     mem.read(raddr, ren && !(circbuffer.io.dataOut.wen)).asUInt
//   } else {
//     mem.read(raddr, ren).asUInt
//   }

//   val fromDMA = circbuffer.io.dataOut.fromDMA

//   val delayed_ren = RegNext(ren, false.B)
//   val delayed_fromDMA = RegNext(fromDMA, false.B)
//   val delayed_tag = RegNext(circbuffer.io.dataOut.tag)
//   val delayed_idx = RegNext(circbuffer.io.dataOut.idx, 0.U(log2Ceil(max_in_flight_sram).W))

//   for (i <- 0 until nSharers) {
//     io.in(i).read.resp.valid := delayed_ren && delayed_tag === i.U
//     io.in(i).read.resp.bits.data := rdata
//     io.in(i).read.resp.bits.fromDMA := delayed_fromDMA
//     io.in(i).read.resp.bits.idx := delayed_idx
//   }

//   circbuffer.io.deqReady := true.B

//   val busy = circbuffer.io.nEnqueued > 0.U
//   dontTouch(busy)
// }

// class ExtAccBank[T <: Data](
//   nSharers: Int, n: Int, t: Vec[Vec[T]], acc_singleported: Boolean, acc_latency: Int,
//   buffer_capacity: Int, max_in_flight_sram: Int, enableExWrite: Boolean = true
// )
//   (implicit ev: Arithmetic[T]) extends Module {

//   import ev._
  
//   val io = IO(new Bundle {
//     val in = Vec(nSharers, Flipped(new ExtAccumulatorBankIO(n, t, buffer_capacity, max_in_flight_sram)))
//     val adder = new Bundle {
//       val valid = Output(Bool())
//       val op1 = Output(t.cloneType)
//       val op2 = Output(t.cloneType)
//       val sum = Input(t.cloneType)
//     }
//   })

//   require (acc_latency >= 2)

//   // Writes take precedence within each selected sharer

//   class ExtAccumulatorReqWTag[T <: Data: Arithmetic, U <: Data](nSharers: Int, n: Int, t: Vec[Vec[T]]) extends Bundle {
//     val ren = Bool()
//     val raddr = UInt(log2Ceil(n).W)
//     val full = Bool()
//     val idx = UInt(log2Ceil(max_in_flight_sram).W)
//     val wen = Bool()
//     val waddr = UInt(log2Ceil(n).W)
//     val wdata = t.cloneType
//     val wmask = Vec(t.getWidth / 8, Bool()) // TODO Use aligned_to here
//     val acc = Bool()
//     val tag = UInt(log2Ceil(nSharers).W)
//     val fromDMA = Bool()
//   }
  
//   val circbuffer = Module(new CircularBuffer(new ExtAccumulatorReqWTag(nSharers, n, t), nSharers, buffer_capacity))
//   val request_accepted = io.in.map{ case wrp => wrp.write.fire || wrp.read.req.fire}
//   val request_accepted_vec = VecInit(request_accepted)
//   val reqRrPtr = RegInit(0.U((1 max log2Ceil(nSharers)).W))

//   circbuffer.io.enqValid := PopCount(request_accepted)
//   circbuffer.io.flush := false.B
//   io.in.foreach(_.nSpace := circbuffer.io.nSpace)

//   private val reqCountBits = log2Ceil(nSharers + 1)
//   private val comingWindow = buffer_capacity
//   private val comingHBits = 1 max log2Ceil((comingWindow max 1) + 1)
//   private val defaultComingH = (comingWindow - 1) max 0

//   val grantedCount = Wire(UInt(reqCountBits.W))
//   val remindedCount = Wire(UInt(reqCountBits.W))
//   val nearestComingH = Wire(UInt(comingHBits.W))
//   nearestComingH := defaultComingH.U

//   if (enableExWrite) {
//     val gSpace = RegInit(buffer_capacity.U(log2Ceil(buffer_capacity + 1).W))
//     val comingRegs = RegInit(VecInit(Seq.fill(comingWindow)(0.U(reqCountBits.W))))
//     val comingPtr = RegInit(0.U((1 max log2Ceil(comingWindow)).W))
//     val grantRrPtr = RegInit(0.U((1 max log2Ceil(nSharers)).W))

//     val availableGrant = Mux(gSpace >= nSharers.U, nSharers.U, gSpace)

//     val grantReqMask = VecInit(io.in.map(_.grant.valid)).asUInt
//     val grantReqMaskRR = grantReqMask.rotateRight(grantRrPtr)
//     val grantCandRemaining = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//     val grantSelectedMask = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//     val grantSelectedCount = Wire(Vec(nSharers + 1, UInt(reqCountBits.W)))
//     grantCandRemaining(0) := grantReqMaskRR
//     grantSelectedMask(0) := 0.U
//     grantSelectedCount(0) := 0.U
//     for (i <- 0 until nSharers) {
//       val pickOH = PriorityEncoderOH(grantCandRemaining(i))
//       val canGrant = pickOH.orR && (grantSelectedCount(i) < availableGrant)
//       val grantOH = Mux(canGrant, pickOH, 0.U(nSharers.W))
//       grantSelectedMask(i + 1) := grantSelectedMask(i) | grantOH
//       grantCandRemaining(i + 1) := grantCandRemaining(i) & ~grantOH
//       grantSelectedCount(i + 1) := grantSelectedCount(i) + canGrant
//     }
//     val grantSelectedMaskRR = grantSelectedMask(nSharers)
//     val grantSelectedMaskOrig = grantSelectedMaskRR.rotateLeft(grantRrPtr)
//     val grantSelected = grantSelectedMaskOrig.asBools

//     for (i <- 0 until nSharers) {
//       io.in(i).grant.ready := grantSelected(i)
//       io.in(i).remind.ready := true.B
//     }

//     grantedCount := PopCount(VecInit(io.in.map(_.grant.fire)))
//     remindedCount := PopCount(VecInit(io.in.map(_.remind.fire)))

//     val rrAdvance = Mux(grantedCount === nSharers.U, 0.U, grantedCount)

//     when (grantedCount =/= 0.U) {
//       grantRrPtr := wrappingAdd(grantRrPtr, rrAdvance, nSharers)
//     }

//     if (comingWindow > 1) {
//       val hFound = Wire(Bool())
//       hFound := false.B
//       nearestComingH := defaultComingH.U
//       for (h <- 1 until comingWindow) {
//         val idx = wrappingAdd(comingPtr, h.U, comingWindow)
//         when (!hFound && comingRegs(idx) =/= 0.U) {
//           nearestComingH := h.U
//           hFound := true.B
//         }
//       }
//     }

//     when (gSpace === buffer_capacity.U) {
//       gSpace := gSpace - grantedCount
//     }.otherwise {
//       gSpace := gSpace - grantedCount + 1.U
//     }
//     comingRegs(comingPtr) := remindedCount
//     comingPtr := wrappingAdd(comingPtr, 1.U, comingWindow)

//     when (reset.asBool) {
//       gSpace := buffer_capacity.U
//     }
//   } else {
//     for (i <- 0 until nSharers) {
//       io.in(i).grant.ready := true.B
//       io.in(i).remind.ready := true.B
//     }
//     grantedCount := 0.U
//     remindedCount := 0.U
//     nearestComingH := defaultComingH.U
//   }

//   val admitRawSigned = circbuffer.io.nSpace.zext + nearestComingH.zext + 1.S - buffer_capacity.S
//   val admitRaw = Mux(admitRawSigned > 0.S, admitRawSigned.asUInt, 0.U)
//   val availableAdmit = Mux(admitRaw >= nSharers.U, nSharers.U, admitRaw)

//   val exwriteMask = VecInit(io.in.map(in => in.write.valid && in.write.bits.exwrite)).asUInt
//   assert(PopCount(exwriteMask) <= 1.U)
//   val hasExwrite = exwriteMask.orR

//   val nonExWriteCandidate = VecInit(io.in.map(in => in.write.valid && !in.write.bits.exwrite && !hasExwrite))
//   val readCandidate = VecInit(io.in.map(in =>
//     in.read.req.valid && !hasExwrite && !(in.write.valid && acc_singleported.B) && !(in.write.valid && in.write.bits.acc)
//   ))
//   val nonExCandidateMask = VecInit(nonExWriteCandidate.zip(readCandidate).map { case (wC, rC) => wC || rC }).asUInt
//   val nonExCandidateMaskRR = nonExCandidateMask.rotateRight(reqRrPtr)

//   val nonExCandRemaining = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   val nonExSelectedMask = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   val nonExSelectedCount = Wire(Vec(nSharers + 1, UInt(reqCountBits.W)))
//   nonExCandRemaining(0) := nonExCandidateMaskRR
//   nonExSelectedMask(0) := 0.U
//   nonExSelectedCount(0) := 0.U
//   for (i <- 0 until nSharers) {
//     val pickOH = PriorityEncoderOH(nonExCandRemaining(i))
//     val canGrant = pickOH.orR && (nonExSelectedCount(i) < availableAdmit)
//     val grantOH = Mux(canGrant, pickOH, 0.U(nSharers.W))
//     nonExSelectedMask(i + 1) := nonExSelectedMask(i) | grantOH
//     nonExCandRemaining(i + 1) := nonExCandRemaining(i) & ~grantOH
//     nonExSelectedCount(i + 1) := nonExSelectedCount(i) + canGrant
//   }

//   val nonExSelectedMaskRR = nonExSelectedMask(nSharers)
//   val nonExSelectedMaskOrig = nonExSelectedMaskRR.rotateLeft(reqRrPtr)
//   val nonExSelected = nonExSelectedMaskOrig.asBools
//   val nonExSelectedTotal = nonExSelectedCount(nSharers)

//   for (i <- 0 until nSharers) {
//     val canTakeNonExWrite = nonExSelected(i) && nonExWriteCandidate(i)
//     val canTakeRead = nonExSelected(i) && readCandidate(i)
//     io.in(i).write.ready := exwriteMask(i) || canTakeNonExWrite
//     io.in(i).read.req.ready := canTakeRead
//   }

//   val writeFireVec = VecInit(io.in.map(_.write.fire))
//   val readFireVec = VecInit(io.in.map(_.read.req.fire))
//   val anyWriteFire = writeFireVec.asUInt.orR
//   val anyReadFire = readFireVec.asUInt.orR

//   when (!hasExwrite && nonExSelectedTotal =/= 0.U) {
//     val rrAdvance = Mux(nonExSelectedTotal === nSharers.U, 0.U, nonExSelectedTotal)
//     reqRrPtr := wrappingAdd(reqRrPtr, rrAdvance, nSharers)
//   }
//   val enqCandidates = Wire(Vec(nSharers, new ExtAccumulatorReqWTag(nSharers, n, t)))
//   for (i <- 0 until nSharers) {
//     enqCandidates(i).ren := io.in(i).read.req.fire
//     enqCandidates(i).raddr := io.in(i).read.req.bits.addr
//     enqCandidates(i).full := io.in(i).read.req.bits.full
//     enqCandidates(i).idx := io.in(i).read.req.bits.idx
//     enqCandidates(i).wen := io.in(i).write.fire
//     enqCandidates(i).waddr := io.in(i).write.bits.addr
//     enqCandidates(i).wdata := io.in(i).write.bits.data
//     enqCandidates(i).wmask := io.in(i).write.bits.mask
//     enqCandidates(i).acc := io.in(i).write.bits.acc
//     enqCandidates(i).fromDMA := io.in(i).read.req.bits.fromDMA
//     enqCandidates(i).tag := i.U
//   }

//   val remainingAccepted = Wire(Vec(nSharers + 1, UInt(nSharers.W)))
//   remainingAccepted(0) := request_accepted_vec.asUInt
//   for (i <- 0 until nSharers) {
//     val pickOH = PriorityEncoderOH(remainingAccepted(i))
//     val pickedReq = Mux1H(pickOH.asBools.zip(enqCandidates))
//     circbuffer.io.enqData(i) := Mux(pickOH.orR, pickedReq, 0.U.asTypeOf(enqCandidates(i)))
//     remainingAccepted(i + 1) := remainingAccepted(i) & ~pickOH
//   }

//   val pipelined_writes = Reg(Vec(acc_latency, Valid(new ExtAccumulatorWriteReq(n, t))))
//   val oldest_pipelined_write = pipelined_writes(acc_latency-1)
//   pipelined_writes(0).valid := circbuffer.io.deqFire() && circbuffer.io.dataOut.wen
//   pipelined_writes(0).bits.addr := circbuffer.io.dataOut.waddr
//   pipelined_writes(0).bits.acc := circbuffer.io.dataOut.acc
//   pipelined_writes(0).bits.data := circbuffer.io.dataOut.wdata
//   pipelined_writes(0).bits.mask := circbuffer.io.dataOut.wmask
//   pipelined_writes(0).bits.exwrite  := DontCare
//   for (i <- 1 until acc_latency) {
//     pipelined_writes(i) := pipelined_writes(i-1)
//   }

//   val rdata_for_adder = Wire(t)
//   rdata_for_adder := DontCare
//   val rdata_for_read_resp = Wire(t)
//   rdata_for_read_resp := DontCare

//   val adder_sum = io.adder.sum
//   io.adder.valid := pipelined_writes(0).valid && pipelined_writes(0).bits.acc
//   io.adder.op1 := rdata_for_adder
//   io.adder.op2 := pipelined_writes(0).bits.data

//   val block_read_req = WireInit(false.B)
//   val block_write_req = WireInit(false.B)

//   val mask_len = t.getWidth / 8
//   val mask_elem = UInt((t.getWidth / mask_len).W)

//   val rvalid = WireInit(false.B)

//   if (!acc_singleported) {
//     val mem = TwoPortSyncMem(n, t, mask_len) // TODO We assume byte-alignment here. Use aligned_to instead
//     mem.io.waddr := oldest_pipelined_write.bits.addr
//     mem.io.wen := oldest_pipelined_write.valid
//     mem.io.wdata := Mux(oldest_pipelined_write.bits.acc, adder_sum, oldest_pipelined_write.bits.data)
//     mem.io.mask := oldest_pipelined_write.bits.mask
//     rdata_for_adder := mem.io.rdata
//     rdata_for_read_resp := mem.io.rdata
//     mem.io.raddr := Mux(circbuffer.io.deqFire() && circbuffer.io.dataOut.acc && circbuffer.io.dataOut.wen, circbuffer.io.dataOut.waddr, circbuffer.io.dataOut.raddr)
//     mem.io.ren := circbuffer.io.deqFire() && (circbuffer.io.dataOut.wen && circbuffer.io.dataOut.acc || circbuffer.io.dataOut.ren)
//   }

//   val delayed_read_valid = RegNext(circbuffer.io.deqFire() && circbuffer.io.dataOut.ren, false.B)
//   val delayed_idx = RegNext(circbuffer.io.dataOut.idx, 0.U(log2Ceil(max_in_flight_sram).W))
//   val delayed_tag = RegNext(circbuffer.io.dataOut.tag)
//   val delayed_fromDMA = RegNext(circbuffer.io.dataOut.fromDMA)

//   for (i <- 0 until nSharers) {
//     io.in(i).read.resp.valid := delayed_read_valid && delayed_tag === i.U
//     io.in(i).read.resp.bits.data := rdata_for_read_resp
//     io.in(i).read.resp.bits.acc_bank_id := DontCare
//     io.in(i).read.resp.bits.idx := delayed_idx
//     io.in(i).read.resp.bits.fromDMA := delayed_fromDMA
//   }

//   // circbuffer.io.deqReady := (selectedQueueWillBeEmpty && circbuffer.io.dataOut.ren) || (circbuffer.io.dataOut.wen && !pipelined_writes(0).valid)
//   circbuffer.io.deqReady := true.B

//   // io.read.req.ready := q_will_be_empty && (
//   //     !pipelined_writes.map(r => r.valid && r.bits.addr === io.read.req.bits.addr).reduce(_||_)  &&
//   //     !block_read_req
//   // )

//   // io.write.ready := !block_write_req &&
//   //   !pipelined_writes.map(r => r.valid && r.bits.addr === io.write.bits.addr && io.write.bits.acc).reduce(_||_)

//   when (reset.asBool) {
//     pipelined_writes.foreach(_.valid := false.B)
//   }

//   val busy = circbuffer.io.nEnqueued > 0.U
//   dontTouch(busy)

//   // assert(!(io.read.req.valid && io.write.en && io.write.acc), "reading and accumulating simultaneously is not supported")
//   // assert(!(io.read.req.fire && io.write.fire && io.read.req.bits.addr === io.write.bits.addr), "reading from and writing to same address is not supported")
// }


class ExtScratchpadBank(
  nSharers: Int, n: Int, w: Int, aligned_to: Int, single_ported: Boolean,
  enableExWrite: Boolean = true
) extends Module {
  require(w % aligned_to == 0 || w < aligned_to)
  val mask_len = (w / (aligned_to * 8)) max 1
  val mask_elem = UInt((w min (aligned_to * 8)).W)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtScratchpadBankIO(n, w, mask_len)))
  })

  val rrPtrWidth = 1 max log2Ceil(nSharers)
  val reqRrPtr = RegInit(0.U(rrPtrWidth.W))
  val tagWidth = 1 max log2Ceil(nSharers)

  if (enableExWrite) {
    val grantRrPtr = RegInit(0.U(rrPtrWidth.W))

    val grantReqMask = VecInit(io.in.map(_.grant.valid)).asUInt
    val grantReqMaskRR = grantReqMask.rotateRight(grantRrPtr)
    val grantSelectedMaskRR = PriorityEncoderOH(grantReqMaskRR)
    val grantSelectedMask = Mux(grantReqMask.orR, grantSelectedMaskRR.rotateLeft(grantRrPtr), 0.U(nSharers.W))
    val grantFireMask = VecInit(io.in.map(_.grant.fire)).asUInt

    for (i <- 0 until nSharers) {
      io.in(i).grant.ready := grantSelectedMask(i)
    }

    when (grantFireMask.orR) {
      val grantWinner = OHToUInt(grantFireMask)
      grantRrPtr := wrappingAdd(grantWinner, 1.U, nSharers)
      // grantRrPtr := wrappingAdd(grantRrPtr, 1.U, nSharers)
    }
  } else {
    for (i <- 0 until nSharers) {
      io.in(i).grant.ready := true.B
    }
  }

  // val singleportBusyWithWrite = io.in.map(_.write.valid).reduce(_||_) && single_ported.B
  val exwriteMask = VecInit(io.in.map(in => in.write.valid && in.write.bits.exwrite)).asUInt
  assert(PopCount(exwriteMask) <= 1.U)
  val hasExwrite = exwriteMask.orR

  val nonExWriteCandidate = VecInit(io.in.map(in => in.write.valid && !in.write.bits.exwrite && !hasExwrite))
  val readCandidate = VecInit(io.in.map(in => in.read.req.valid && !hasExwrite && !(in.write.valid && single_ported.B)))
  val nonExCandidateMask = VecInit(nonExWriteCandidate.zip(readCandidate).map { case (wC, rC) => wC || rC }).asUInt
  val nonExCandidateMaskRR = nonExCandidateMask.rotateRight(reqRrPtr)
  val nonExSelectedMaskRR = PriorityEncoderOH(nonExCandidateMaskRR)
  val nonExSelectedMask = Mux(nonExCandidateMask.orR, nonExSelectedMaskRR.rotateLeft(reqRrPtr), 0.U(nSharers.W))
  val nonExSelected = nonExSelectedMask.asBools

  for (i <- 0 until nSharers) {
    val canTakeNonExWrite = nonExSelected(i) && nonExWriteCandidate(i)
    val canTakeRead = nonExSelected(i) && readCandidate(i)
    io.in(i).write.ready := exwriteMask(i) || canTakeNonExWrite
    io.in(i).read.req.ready := canTakeRead
  }

  val writeFireVec = VecInit(io.in.map(_.write.fire))
  val readFireVec = VecInit(io.in.map(_.read.req.fire))
  val reqFireMask = writeFireVec.asUInt | readFireVec.asUInt
  val anyWriteFire = writeFireVec.asUInt.orR
  val anyReadFire = readFireVec.asUInt.orR

  when (!hasExwrite && reqFireMask.orR) {
    val reqWinner = OHToUInt(reqFireMask)
    reqRrPtr := wrappingAdd(reqWinner, 1.U, nSharers)
    // reqRrPtr := wrappingAdd(reqRrPtr, 1.U, nSharers)
  }

  val selectedWaddr = Mux1H(writeFireVec, io.in.map(_.write.bits.addr))
  val selectedWdata = Mux1H(writeFireVec, io.in.map(_.write.bits.data))
  val selectedWmask = Mux1H(writeFireVec, io.in.map(_.write.bits.mask))
  val selectedRaddr = Mux1H(readFireVec, io.in.map(_.read.req.bits.addr))
  val selectedTag = Mux1H(readFireVec, (0 until nSharers).map(i => i.U(tagWidth.W)))

  val mem = SyncReadMem(n, Vec(mask_len, UInt(mask_elem.getWidth.W)))
  when (anyWriteFire) {
    if (aligned_to >= w) {
      mem.write(selectedWaddr, selectedWdata.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
    } else {
      mem.write(selectedWaddr, selectedWdata.asTypeOf(Vec(mask_len, mask_elem)), selectedWmask)
    }
  }

  val ren = anyReadFire
  val rdata = if (single_ported) {
    assert(!(ren && anyWriteFire))
    mem.read(selectedRaddr, ren && !anyWriteFire).asUInt
  } else {
    mem.read(selectedRaddr, ren).asUInt
  }

  val delayedRen = RegNext(ren, false.B)
  val delayedTag = RegNext(selectedTag)

  for (i <- 0 until nSharers) {
    io.in(i).read.resp.valid := delayedRen && delayedTag === i.U(tagWidth.W)
    io.in(i).read.resp.bits.data := rdata
  }
}

class ExtAccBank[T <: Data](
  nSharers: Int, n: Int, t: Vec[Vec[T]], accBanks: Int,
  acc_singleported: Boolean, acc_latency: Int,
  enableExWrite: Boolean = true, atomicClient: Int = -1
)
  (implicit ev: Arithmetic[T]) extends Module {

  import ev._

  require(atomicClient < nSharers)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtAccumulatorBankIO(
      n, t, accBanks)))
    val atomic = if (atomicClient >= 0) {
      Some(Flipped(new ExtAccumulatorAtomicClientIO))
    } else None
    val adder = new Bundle {
      val valid = Output(Bool())
      val op1 = Output(t.cloneType)
      val op2 = Output(t.cloneType)
      val sum = Input(t.cloneType)
    }
  })

  require(acc_latency >= 2)

  val rrPtrWidth = 1 max log2Ceil(nSharers)
  val grantRrPtr = RegInit(0.U(rrPtrWidth.W))
  val reqRrPtr = RegInit(0.U(rrPtrWidth.W))
  val tagWidth = 1 max log2Ceil(nSharers)

  if (enableExWrite) {
    val grantReqMask = VecInit(io.in.map(_.grant.valid)).asUInt
    val grantReqMaskRR = grantReqMask.rotateRight(grantRrPtr)
    val grantSelectedMaskRR = PriorityEncoderOH(grantReqMaskRR)
    val grantSelectedMask = Mux(grantReqMask.orR, grantSelectedMaskRR.rotateLeft(grantRrPtr), 0.U(nSharers.W))
    val grantFireMask = VecInit(io.in.map(_.grant.fire)).asUInt

    for (i <- 0 until nSharers) {
      io.in(i).grant.ready := grantSelectedMask(i)
    }

    when (grantFireMask.orR) {
      val grantWinner = OHToUInt(grantFireMask)
      grantRrPtr := wrappingAdd(grantWinner, 1.U, nSharers)
      // grantRrPtr := wrappingAdd(grantRrPtr, 1.U, nSharers)
    }
  } else {
    for (i <- 0 until nSharers) {
      io.in(i).grant.ready := true.B
    }
  }

  // val singleportBusyWithWrite = io.in.map(_.write.valid).reduce(_||_) && acc_singleported.B
  // val readportBusyWithAccRead = io.in.map(in => in.write.valid && in.write.bits.acc).reduce(_||_)

  val exwriteMask = VecInit(io.in.map(in => in.write.valid && in.write.bits.exwrite)).asUInt
  assert(PopCount(exwriteMask) <= 1.U)
  val hasExwrite = exwriteMask.orR

  val pipelined_writes = Reg(Vec(acc_latency, Valid(new ExtAccumulatorWriteReq(
    n, t))))

  val nonExWriteCandidate = VecInit(io.in.map(in => in.write.valid && !in.write.bits.exwrite && !hasExwrite && !pipelined_writes.map(r => r.valid && r.bits.addr === in.write.bits.addr && in.write.bits.acc).reduce(_||_)))
  val readCandidate = VecInit(io.in.map(in =>
    in.read.req.valid && !hasExwrite && !(in.write.valid && acc_singleported.B) && !(in.write.valid && in.write.bits.acc) &&
      !pipelined_writes.map(r => r.valid && r.bits.addr === in.read.req.bits.addr).reduce(_||_)
  ))
  val nonExCandidateMask = VecInit(nonExWriteCandidate.zip(readCandidate).map { case (wC, rC) => wC || rC }).asUInt
  val nonExCandidateMaskRR = nonExCandidateMask.rotateRight(reqRrPtr)
  val nonExSelectedMaskRR = PriorityEncoderOH(nonExCandidateMaskRR)
  val nonExSelectedMask = Mux(nonExCandidateMask.orR, nonExSelectedMaskRR.rotateLeft(reqRrPtr), 0.U(nSharers.W))
  val nonExSelected = nonExSelectedMask.asBools

  for (i <- 0 until nSharers) {
    val canTakeNonExWrite = nonExSelected(i) && nonExWriteCandidate(i)
    val canTakeRead = nonExSelected(i) && readCandidate(i)
    val writeAllowed = if (i == atomicClient) {
      io.atomic.get.writeAllow
    } else true.B
    val readAllowed = if (i == atomicClient) {
      io.atomic.get.readAllow
    } else true.B
    io.in(i).write.ready := exwriteMask(i) ||
      (canTakeNonExWrite && writeAllowed)
    io.in(i).read.req.ready := canTakeRead && readAllowed
  }

  if (atomicClient >= 0) {
    io.atomic.get.writeSelected :=
      nonExSelected(atomicClient) && nonExWriteCandidate(atomicClient)
    io.atomic.get.readSelected :=
      nonExSelected(atomicClient) && readCandidate(atomicClient)
  }

  val writeFireVec = VecInit(io.in.map(_.write.fire))
  val readFireVec = VecInit(io.in.map(_.read.req.fire))
  val reqFireMask = writeFireVec.asUInt | readFireVec.asUInt
  val anyWriteFire = writeFireVec.asUInt.orR
  val anyReadFire = readFireVec.asUInt.orR

  when (!hasExwrite && reqFireMask.orR) {
    val reqWinner = OHToUInt(reqFireMask)
    reqRrPtr := wrappingAdd(reqWinner, 1.U, nSharers)
    // reqRrPtr := wrappingAdd(reqRrPtr, 1.U, nSharers)
  }

  val selectedWriteAddr = Mux1H(writeFireVec, io.in.map(_.write.bits.addr))
  val selectedWriteData = Mux1H(writeFireVec, io.in.map(_.write.bits.data))
  val selectedWriteMask = Mux1H(writeFireVec, io.in.map(_.write.bits.mask))
  val selectedWriteAcc = Mux1H(writeFireVec, io.in.map(_.write.bits.acc))
  val selectedWriteExwrite = Mux1H(writeFireVec, io.in.map(_.write.bits.exwrite))
  val selectedReadAddr = Mux1H(readFireVec, io.in.map(_.read.req.bits.addr))
  val selectedReadTag = Mux1H(readFireVec, (0 until nSharers).map(i => i.U(tagWidth.W)))

  val oldest_pipelined_write = pipelined_writes(acc_latency - 1)
  pipelined_writes(0).valid := anyWriteFire
  pipelined_writes(0).bits.addr := selectedWriteAddr
  pipelined_writes(0).bits.acc := selectedWriteAcc
  pipelined_writes(0).bits.data := selectedWriteData
  pipelined_writes(0).bits.mask := selectedWriteMask
  pipelined_writes(0).bits.exwrite := selectedWriteExwrite
  for (i <- 1 until acc_latency) {
    pipelined_writes(i) := pipelined_writes(i - 1)
  }

  val rdata_for_adder = Wire(t)
  rdata_for_adder := 0.U.asTypeOf(t)
  val rdata_for_read_resp = Wire(t)
  rdata_for_read_resp := 0.U.asTypeOf(t)

  val adder_sum = io.adder.sum
  io.adder.valid := pipelined_writes(0).valid && pipelined_writes(0).bits.acc
  io.adder.op1 := rdata_for_adder
  io.adder.op2 := pipelined_writes(0).bits.data

  val mask_len = t.getWidth / 8
  val mem = TwoPortSyncMem(n, t, mask_len)
  mem.io.waddr := oldest_pipelined_write.bits.addr
  val finalWriteData = Mux(oldest_pipelined_write.bits.acc,
    adder_sum, oldest_pipelined_write.bits.data)
  mem.io.wen := oldest_pipelined_write.valid
  mem.io.wdata := finalWriteData
  mem.io.mask := oldest_pipelined_write.bits.mask
  mem.io.raddr := Mux(anyWriteFire && selectedWriteAcc, selectedWriteAddr, selectedReadAddr)
  mem.io.ren := (anyWriteFire && selectedWriteAcc) || anyReadFire
  rdata_for_adder := mem.io.rdata
  rdata_for_read_resp := mem.io.rdata

  val delayed_read_valid = RegNext(anyReadFire, false.B)
  val delayed_tag = RegNext(selectedReadTag)

  for (i <- 0 until nSharers) {
    io.in(i).read.resp.valid := delayed_read_valid && delayed_tag === i.U(tagWidth.W)
    io.in(i).read.resp.bits.data := rdata_for_read_resp
    io.in(i).read.resp.bits.acc_bank_id := DontCare
  }

  when (reset.asBool) {
    pipelined_writes.foreach(_.valid := false.B)
  }
}

class SharedSyncReadMem(nSharers: Int, depth: Int, mask_len: Int, data_len: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtMemIO()))
  })
  val mem = SyncReadMem(depth, Vec(mask_len, UInt(data_len.W)))
  val wens = io.in.map(_.write_en)
  val wen = wens.reduce(_||_)
  val waddr = Mux1H(wens, io.in.map(_.write_addr))
  val wmask = Mux1H(wens, io.in.map(_.write_mask))
  val wdata = Mux1H(wens, io.in.map(_.write_data))
  assert(PopCount(wens) <= 1.U)
  val rens = io.in.map(_.read_en)
  assert(PopCount(rens) <= 1.U)
  val ren = rens.reduce(_||_)
  val raddr = Mux1H(rens, io.in.map(_.read_addr))
  val rdata = mem.read(raddr, ren && !wen)
  io.in.foreach(_.read_data := rdata.asUInt)
  when (wen) {
    mem.write(waddr, wdata.asTypeOf(Vec(mask_len, UInt(data_len.W))), wmask.asTypeOf(Vec(mask_len, Bool())))
  }

}

// // made
// class SharedSyncReadMem_4(nSharers: Int, depth: Int, mask_len: Int, data_len: Int) extends Module {
//   val io = IO(new Bundle {
//     val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
//   })

//   // val mem = SRAM(depth, Vec(mask_len, UInt(data_len.W)), 0, 0, 4, true)
//   val mem = SRAM.masked(depth, Vec(mask_len, UInt(data_len.W)), 4, 4, 0)

//   for (i <- 0 until nSharers) {
//     mem.readPorts(i).enable := false.B
//     mem.writePorts(i).enable := false.B
//     when(io.in(i).read_en) {
//       mem.readPorts(i).enable := true.B
//     }.elsewhen(io.in(i).write_en) {
//       mem.writePorts(i).enable := true.B
//     }

//     mem.readPorts(i).address := io.in(i).read_addr
//     mem.writePorts(i).address := io.in(i).write_addr

//     mem.writePorts(i).data := io.in(i).write_data.asTypeOf(Vec(mask_len, UInt(data_len.W)))
//     val rdata = WireInit(Vec(mask_len, UInt(data_len.W)), mem.readPorts(i).data)
//     io.in(i).read_data := rdata.asUInt
    
//     mem.writePorts(i).mask.foreach { m =>
//       m := io.in(i).write_mask.asTypeOf(Vec(mask_len, Bool()))
//     }
//   }
// }

// /*
//  * LVT-based 4R4W memory using SyncReadMem.
//  */
// class SharedSyncReadMem_4_LVT(
//   nSharers: Int,
//   depth: Int,
//   mask_len: Int,
//   data_len: Int
// ) extends Module {
//   val io = IO(new Bundle {
//     val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
//   })

//   val W = nSharers
//   val R = nSharers
//   val addrWidth = log2Ceil(depth)
//   val writerBits = log2Ceil(W)

//   // 1. Data Replicas (BRAM 기반)
//   val mem = Seq.fill(W, R) {
//     SyncReadMem(depth, Vec(mask_len, UInt(data_len.W)))
//   }

//   // 2. LVT (LUTRAM 기반 선호)
//   // Reg 대신 Mem을 사용하여 자원 절약. asyncRead가 가능한 Mem 사용 시 sel 타이밍 유지 가능
//   val lvt = Mem(depth, UInt(writerBits.W))

//   // --- Write Logic ---
//   for (i <- 0 until W) {
//     val wPort = io.in(i)
//     when(wPort.write_en) {
//       lvt.write(wPort.write_addr, i.U) // 누가 썼는지 기록
//       for (r <- 0 until R) {
//         mem(i)(r).write(wPort.write_addr, 
//                              wPort.write_data.asTypeOf(Vec(mask_len, UInt(data_len.W))), 
//                              wPort.write_mask.asTypeOf(Vec(mask_len, Bool())))
//       }
//     }
//   }

//   // --- Read Logic ---
//   for (r <- 0 until R) {
//     val rPort = io.in(r)
//     // 읽기 활성화 시 현재 주소의 최신 작성자 index 확인
//     // Mem의 read(addr)은 조합회로(Async)처럼 동작하거나 동기식으로 설정 가능
//     val sel = lvt.read(rPort.read_addr) 
//     val sel_p1 = RegNext(sel)
    
//     val cand = Wire(Vec(W, Vec(mask_len, UInt(data_len.W))))
//     for (w <- 0 until W) {
//       // SyncReadMem은 t+1 시점에 데이터를 출력
//       cand(w) := mem(w)(r).read(rPort.read_addr, rPort.read_en)
//     }

//     // t+1 시점에 최신 작성자 데이터를 선택
//     val chosen = cand(sel_p1)
//     rPort.read_data := chosen.asUInt
//   }
// }

// made end

class SharedExtMem(
  sp_banks: Int, acc_banks: Int, acc_sub_banks: Int,
  sp_depth: Int, sp_mask_len: Int, sp_data_len: Int,
  acc_depth: Int, acc_mask_len: Int, acc_data_len: Int
) extends Module {
  val nSharers = 2
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtSpadMemIO(sp_banks, acc_banks, acc_sub_banks)))
  })
  for (i <- 0 until sp_banks) {
    val spad_mem = Module(new SharedSyncReadMem(nSharers, sp_depth, sp_mask_len, sp_data_len))
    for (w <- 0 until nSharers) {
      spad_mem.io.in(w) <> io.in(w).spad(i)
    }
  }
  for (i <- 0 until acc_banks) {
    for (s <- 0 until acc_sub_banks) {
      val acc_mem = Module(new SharedSyncReadMem(nSharers, acc_depth, acc_mask_len, acc_data_len))

      acc_mem.io.in(0) <> io.in(0).acc(i)(s)
      // The FP gemmini expects a taller, skinnier accumulator mem
      acc_mem.io.in(1) <> io.in(1).acc(i)(s)
      acc_mem.io.in(1).read_addr := io.in(1).acc(i)(s).read_addr >> 1
      io.in(1).acc(i)(s).read_data := acc_mem.io.in(1).read_data.asTypeOf(Vec(2, UInt((acc_data_len * acc_mask_len / 2).W)))(RegNext(io.in(1).acc(i)(s).read_addr(0)))

      acc_mem.io.in(1).write_addr := io.in(1).acc(i)(s).write_addr >> 1
      acc_mem.io.in(1).write_data := Cat(io.in(1).acc(i)(s).write_data, io.in(1).acc(i)(s).write_data)
      acc_mem.io.in(1).write_mask := Mux(io.in(1).acc(i)(s).write_addr(0), io.in(1).acc(i)(s).write_mask << (acc_mask_len / 2), io.in(1).acc(i)(s).write_mask)
    }
  }
}

// made
class SharedExtMem_4[T <: Data, U <: Data, V <: Data](
    config: GemminiArrayConfig[T, U, V],
    vpuLanes: Int = 16,
    vpuTagBits: Int = 1)
    (implicit p: Parameters, ev: Arithmetic[T]) extends Module {
  
  import config._

  val block_cols = meshColumns * tileColumns
  val spad_w = inputType.getWidth *  block_cols
  val sp_mask_len = (spad_w / (aligned_to * 8)) max 1
  val acc_row_t = Vec(meshColumns, Vec(tileColumns, accType))
  val vpuElementAddrBits = 1 max log2Ceil(
    acc_banks * acc_bank_entries * block_cols)
  val accClients = nSharers + (if (use_vpu_fusion) 1 else 0)
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtMemIO_new(
      sp_banks, sp_sub_banks, sp_bank_entries / sp_sub_banks, spad_w, sp_mask_len,
      acc_banks, acc_sub_banks, acc_bank_entries / acc_sub_banks, acc_row_t
    )))
    val vpuMemory = if (use_vpu_fusion) Some(Flipped(
      new GemminiVpuMemoryIO(
        vpuElementAddrBits, vpuLanes, accType.getWidth, vpuTagBits))) else None
  })

  for (i <- 0 until sp_banks) {
    for (s <- 0 until sp_sub_banks) {
      val spad_mem = Module(new ExtScratchpadBank(
        nSharers, sp_bank_entries / sp_sub_banks, spad_w, aligned_to, sp_singleported,
        ex_write_to_spad
      ))
      for (w <- 0 until nSharers) {
        spad_mem.io.in(w) <> io.in(w).spad(i)(s)
      }
    }
  }

  val acc_mems = Seq.tabulate(acc_banks, acc_sub_banks) { (i, s) =>
    val acc_mem = Module(new ExtAccBank(
      accClients, acc_bank_entries / acc_sub_banks, acc_row_t,
      acc_banks, acc_singleported, acc_latency, ex_write_to_acc,
      atomicClient = if (use_vpu_fusion) nSharers else -1
    ))

    for (w <- 0 until nSharers) {
      acc_mem.io.in(w) <> io.in(w).acc(i)(s)
    }

    acc_mem
  }

  if (use_vpu_fusion) {
    val vpuAdapter = Module(new VpuSharedAccumulatorAdapter(
      config, vpuLanes, vpuTagBits))
    vpuAdapter.io.vpu <> io.vpuMemory.get
    for (bank <- 0 until acc_banks; subBank <- 0 until acc_sub_banks) {
      vpuAdapter.io.memory(bank)(subBank) <>
        acc_mems(bank)(subBank).io.in(nSharers)
      vpuAdapter.io.atomic(bank)(subBank) <>
        acc_mems(bank)(subBank).io.atomic.get
    }
  }

  val flat_acc_mems = acc_mems.flatten
  val nAccAdderReqs = acc_banks * acc_sub_banks

  val adder_req_valid = VecInit(flat_acc_mems.map(_.io.adder.valid))
  val adder_req_op1 = VecInit(flat_acc_mems.map(_.io.adder.op1))
  val adder_req_op2 = VecInit(flat_acc_mems.map(_.io.adder.op2))

  // External issue logic guarantees this, so every valid request can be assigned to a shared adder lane.
  // assert(PopCount(adder_req_valid) <= nSharers.U)

  val selector = Module(new AdderSellector(nSharers, nAccAdderReqs))
  selector.io.in_valid := adder_req_valid

  val acc_adders = Seq.fill(nSharers) {
    Module(new AccPipeShared(acc_latency - 1, acc_row_t, nAccAdderReqs))
  }
  for (a <- 0 until nSharers) {
    acc_adders(a).io.in_sel := VecInit(selector.io.selected_oh(a).asBools)
    acc_adders(a).io.ina := adder_req_op1
    acc_adders(a).io.inb := adder_req_op2
  }

  val adder_sel_oh_d = Seq.tabulate(nSharers) { a =>
    ShiftRegister(selector.io.selected_oh(a), acc_latency - 1)
  }

  for (b <- 0 until nAccAdderReqs) {
    flat_acc_mems(b).io.adder.sum := 0.U.asTypeOf(acc_row_t)

    for (a <- 0 until nSharers) {
      when (adder_sel_oh_d(a)(b)) {
        flat_acc_mems(b).io.adder.sum := acc_adders(a).io.out
      }
    }
  }

  // val selector = Module(new AdderSellector(nAccAdderReqs, nAccAdderReqs))
  // selector.io.in_valid := adder_req_valid

  // val acc_adders = Seq.fill(nAccAdderReqs) {
  //   Module(new AccPipeShared(acc_latency - 1, acc_row_t, nAccAdderReqs))
  // }
  // for (a <- 0 until nAccAdderReqs) {
  //   acc_adders(a).io.in_sel := VecInit(selector.io.selected_oh(a).asBools)
  //   acc_adders(a).io.ina := adder_req_op1
  //   acc_adders(a).io.inb := adder_req_op2
  // }

  // val adder_sel_oh_d = Seq.tabulate(nAccAdderReqs) { a =>
  //   ShiftRegister(selector.io.selected_oh(a), acc_latency - 1)
  // }

  // for (b <- 0 until nAccAdderReqs) {
  //   flat_acc_mems(b).io.adder.sum := 0.U.asTypeOf(acc_row_t)

  //   for (a <- 0 until nAccAdderReqs) {
  //     when (adder_sel_oh_d(a)(b)) {
  //       flat_acc_mems(b).io.adder.sum := acc_adders(a).io.out
  //     }
  //   }
  // }
}
