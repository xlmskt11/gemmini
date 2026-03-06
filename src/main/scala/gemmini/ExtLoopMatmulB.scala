package gemmini

import chisel3._
import chisel3.util._
import Util._

// address range
// class AddressRange(val max_addr: Int) extends Bundle {
//   val start = UInt(log2Up(max_addr).W)
//   val end = UInt(log2Up(max_addr+1).W)

//   def overlaps(other: AddressRange): Bool = {
//     ((other.start <= start && start < other.end) ||
//       (start <= other.start && other.start < end))
//   }
// }

class LdBState(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val group_list = UInt(nSharers.W)
  val k         = UInt(iterator_bitwidth.W)
  val k_offset  = UInt(iterator_bitwidth.W)
  val max_k = UInt(iterator_bitwidth.W)
  val j          = UInt(iterator_bitwidth.W)
  val idle = Bool()
}

class ExState(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val group_list = UInt(nSharers.W)
  val k         = UInt(iterator_bitwidth.W)
  val j          = UInt(iterator_bitwidth.W)
  val idle = Bool()
}

class StCState(
  group_w: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val idle = Bool()
}

class LdBExIO(
  group_w: Int,
  nSharers: Int,
  iterator_bitwidth: Int
) extends Bundle {
  val ldb = Output(new LdBState(group_w, nSharers, iterator_bitwidth))
  val ex = Output(new ExState(group_w, nSharers, iterator_bitwidth))
  val stc = Output(new StCState(group_w, iterator_bitwidth))
  val ldb_ahead    = Input(Bool())
  val loop_full = Input(Bool())
}

class LdBCompleteControl(
  nSharers: Int
) extends Module {
  val iterator_bitwidth = 16
  val concurrent_loops = 2
  val group_num = nSharers * concurrent_loops
  val group_w = log2Up(group_num)

  require(nSharers > 0)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new LdBExIO(group_w, nSharers, iterator_bitwidth)))
  })

  io.in.foreach(_.ldb_ahead := false.B)

  class MemberLdbData extends Bundle {
    val k_offset = UInt(iterator_bitwidth.W)
    val max_k = UInt(iterator_bitwidth.W)
  }

  class MemberData extends Bundle {
    val ldb_end_data = Valid(new MemberLdbData)
    val ex_completed = Bool()
    val stc_completed = Bool()
  }

  class GroupData extends Bundle {
    val mem_data = Vec(nSharers, new MemberData)
    val group_id = UInt(group_w.W)
  }

  val group_data = Reg(Vec(group_num, Valid(new GroupData)))
  val ldb_idle_delayed = RegNext(VecInit(io.in.map(_.ldb.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val ex_idle_delayed  = RegNext(VecInit(io.in.map(_.ex.idle)),  VecInit(Seq.fill(nSharers)(true.B)))
  val stc_idle_delayed = RegNext(VecInit(io.in.map(_.stc.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val weights = RegInit(VecInit(Seq.fill(nSharers)(0.U(log2Ceil(nSharers + 1).W))))

  // Make new group mask
  val groupMask = WireInit(VecInit(Seq.fill(nSharers)(0.U(group_num.W))))
  for (i <- 0 until nSharers) {
    when (!io.in(i).ldb.idle && !(group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)).reduce(_||_))) {
      groupMask(i) := (1.U(group_num.W) << io.in(i).ldb.group_id)
    }
  }
  val reducedMask = groupMask.reduce(_|_)

  // Make weight mask of each group
  val groupWeights = WireInit(VecInit(Seq.fill(group_num)(0.U(log2Ceil(nSharers + 1).W))))
  for (gId <- 0 until group_num) {
    val weightsForThisGroup = (0 until nSharers).map { sIdx =>
      val isMatch = !io.in(sIdx).ldb.idle && io.in(sIdx).ldb.group_id === gId.U
      Mux(isMatch, weights(sIdx), 0.U)
    }
    groupWeights(gId) := weightsForThisGroup.reduce((a, b) => Mux(a > b, a, b))
  }

  // Find higher group number of each group
  val higherGroupNum = Wire(Vec(group_num, UInt(log2Ceil(nSharers + 1).W)))
  for (i <- 0 until group_num) {
    val higherWeightCount = (0 until group_num).map { j =>
      val higher = (groupWeights(j) > groupWeights(i)) || (groupWeights(j) === groupWeights(i) && j.U > i.U)
      reducedMask(j) && higher
    }
    higherGroupNum(i) := PopCount(higherWeightCount)
  }

  // For debug
  val nSpace = PopCount(group_data.map(gd => !gd.valid))
  val inputNum = PopCount(reducedMask)
  dontTouch(nSpace)
  dontTouch(inputNum)

  // Make winners mask
  val winners = (0 until group_num).map(i => reducedMask(i) && (higherGroupNum(i) < nSpace))
  val winnersVec = VecInit(winners)
  val actualInputNum = PopCount(winners)
  dontTouch(actualInputNum)

  // Make loosers mask
  val loosers = (0 until group_num).map(i => reducedMask(i) && !winners(i))
  val failedInputNum = PopCount(loosers)
  dontTouch(failedInputNum)

  // Weight update
  val waitMatrix = RegInit(VecInit(Seq.fill(nSharers)(0.U(nSharers.W))))
  val isWaitingReg = RegInit(0.U(nSharers.W))
  val currentWaitingWire = Wire(Vec(nSharers, Bool()))

  val currentWaitingMask = currentWaitingWire.asUInt
  val leavingWaiters = isWaitingReg & ~currentWaitingMask

  for (i <- 0 until nSharers) {
    val theresMyGroup = group_data.map(gd => 
      gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)
    )
    val onGoing = group_data.zip(theresMyGroup).map{ case (gd, matched) => 
      matched && !gd.bits.mem_data(i).ldb_end_data.valid
    }.reduce(_||_)

    val isWaiting = !io.in(i).ldb.idle && !onGoing
    currentWaitingWire(i) := isWaiting

    val rowWithoutLeavers = waitMatrix(i) & ~leavingWaiters
    val nextRow = Mux(!isWaiting, 0.U(nSharers.W), Mux(!isWaitingReg(i), isWaitingReg & ~leavingWaiters, rowWithoutLeavers))
    waitMatrix(i) := nextRow

    val olderThanMeCount = PopCount(nextRow)
    when (isWaiting) {
      weights(i) := (nSharers.U - 1.U) - olderThanMeCount
    } .otherwise {
      weights(i) := 0.U
    }
  }

  isWaitingReg := currentWaitingMask

  // Allocate groups to free slots
  val winnerMask = winnersVec.asUInt
  val allocMasks = Wire(Vec(group_num, UInt(group_num.W)))
  allocMasks(0) := winnerMask

  for (i <- 0 until group_num) {
    val canAllocate = !group_data(i).valid && allocMasks(i).orR
    val targetId = PriorityEncoder(allocMasks(i))
    
    val winnerGroupList = MuxCase(0.U, (0 until nSharers).map { sIdx =>
      val isMatch = !io.in(sIdx).ldb.idle && io.in(sIdx).ldb.group_id === targetId
      val isTopWeight = weights(sIdx) === groupWeights(targetId)
      (isMatch && isTopWeight) -> io.in(sIdx).ldb.group_list
    })

    when (canAllocate) {
      group_data(i).valid := true.B
      group_data(i).bits.group_id := targetId
      
      val group_mask = VecInit(winnerGroupList.asBools)
      group_data(i).bits.mem_data.zip(group_mask).foreach { case (md, gm) =>
        md.ldb_end_data.valid := !gm
        md.ex_completed := !gm
        md.stc_completed := !gm
        md.ldb_end_data.bits.max_k := 0.U
        md.ldb_end_data.bits.k_offset := 0.U
      }
    }
    
    if (i < group_num - 1) {
      allocMasks(i+1) := Mux(canAllocate, allocMasks(i) & ~(1.U(group_num.W) << targetId), allocMasks(i))
    }
  }

  // Set loop_full signals
  for (i <- 0 until nSharers) {
    val theresMyGroup = group_data.map(gd => 
      gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)
    )
    val onGoing = group_data.zip(theresMyGroup).map{ case (gd, matched) => 
      matched && !gd.bits.mem_data(i).ldb_end_data.valid
    }.reduce(_||_)

    io.in(i).loop_full := !onGoing
  }

  // update group data when members complete
  for (i <- 0 until nSharers) {
    group_data.foreach { gd =>
      when (io.in(i).ldb.idle && !ldb_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)) {
        gd.bits.mem_data(i).ldb_end_data.valid := true.B
        gd.bits.mem_data(i).ldb_end_data.bits.max_k := io.in(i).ldb.max_k
        gd.bits.mem_data(i).ldb_end_data.bits.k_offset := io.in(i).ldb.k_offset
      }

      when (io.in(i).ex.idle && !ex_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        gd.bits.mem_data(i).ex_completed := true.B
      }

      when (io.in(i).stc.idle && !stc_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).stc.group_id)) {
        gd.bits.mem_data(i).stc_completed := true.B
      }
    }
  }

  // if all members have completed, free the group
  group_data.foreach { gd =>
    when (gd.valid && gd.bits.mem_data.map { md => md.ex_completed && md.stc_completed && md.ldb_end_data.valid }.reduce(_&&_)) {
      gd.valid := false.B
      gd.bits.mem_data.foreach { md =>
        md.ex_completed := false.B
        md.stc_completed := false.B
        md.ldb_end_data.valid := false.B
        md.ldb_end_data.bits.k_offset := 0.U
        md.ldb_end_data.bits.max_k := 0.U
      }
      gd.bits.group_id := 0.U
    }
  }

  // logic to set ldb_ahead
  for (i <- 0 until nSharers) {
    val group_list  = io.in(i).ex.group_list
    val group_mask  = VecInit(group_list.asBools)
    val ldb_completed = WireInit(false.B)

    group_data.foreach { gd =>
      when (gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        ldb_completed := gd.bits.mem_data.zip(group_mask).map { case (gdd, gmm) =>
          gmm && gdd.ldb_end_data.valid && (io.in(i).ex.k >= gdd.ldb_end_data.bits.k_offset) && (io.in(i).ex.k < gdd.ldb_end_data.bits.k_offset + gdd.ldb_end_data.bits.max_k)
        }.reduce(_||_)
      }
    }

    val ld_ahead = io.in.zip(group_mask).map{ case (in, m) => m && (io.in(i).ex.group_id === in.ldb.group_id) && (io.in(i).ex.k >= in.ldb.k_offset) && ((in.ldb.k_offset + in.ldb.k > io.in(i).ex.k) || ((in.ldb.k_offset + in.ldb.k === io.in(i).ex.k && in.ldb.j > io.in(i).ex.j)))}.reduce(_||_)

    // io.in(i).ldb_ahead := (ldb_completed || ld_ahead) && !(group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ex.group_id) && gd.bits.mem_data(i).ex_completed).reduce(_||_))
    io.in(i).ldb_ahead := ldb_completed || ld_ahead
  }

  // reset logic
  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
    group_data.foreach { gd =>
      gd.bits.mem_data.foreach(_.ldb_end_data.valid := false.B)
      gd.bits.mem_data.foreach(_.ex_completed := false.B)
      gd.bits.mem_data.foreach(_.stc_completed := false.B)
      gd.bits.mem_data.foreach(_.ldb_end_data.bits.k_offset := 0.U)
      gd.bits.mem_data.foreach(_.ldb_end_data.bits.max_k := 0.U)
    }
    group_data.foreach(_.bits.group_id := 0.U)

    weights.foreach(_ := 0.U)
  }
}
