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

class LdInputState(
  group_w: Int,
  nSharers: Int,
  large_iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val group_list = UInt(nSharers.W)
  val idle = Bool()
}

class ConvExState(
  group_w: Int,
  nSharers: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val group_list = UInt(nSharers.W)
  val idle = Bool()
}

class StState(
  group_w: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
  val idle = Bool()
}

class LdIExIO(
  group_w: Int,
  nSharers: Int,
  large_iterator_bitwidth: Int
) extends Bundle {
  val ldinput = Output(new LdInputState(group_w, nSharers, large_iterator_bitwidth))
  val ex = Output(new ConvExState(group_w, nSharers))
  val st = Output(new StState(group_w))
  val lda_ahead    = Input(Bool())
  val loop_full = Input(Bool())
}

class LdICompleteControl(
  nSharers: Int
) extends Module {
  val large_iterator_bitwidth = 16
  val concurrent_loops = 2
  val group_num = nSharers * concurrent_loops
  val group_w = log2Up(group_num) + 1

  require(nSharers > 0)

  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new LdIExIO(group_w, nSharers, large_iterator_bitwidth)))
  })

  io.in.foreach(_.lda_ahead := false.B)

  class MemberData extends Bundle {
    val ldinput_completed = Bool()
    val ex_completed = Bool()
    val st_completed = Bool()
  }

  class GroupData extends Bundle {
    val mem_data = Vec(nSharers, new MemberData)
    val group_id = UInt(group_w.W)
  }

  val group_data = Reg(Vec(group_num, Valid(new GroupData)))
  val ldinput_idle_delayed = RegNext(VecInit(io.in.map(_.ldinput.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val ex_idle_delayed = RegNext(VecInit(io.in.map(_.ex.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val st_idle_delayed = RegNext(VecInit(io.in.map(_.st.idle)), VecInit(Seq.fill(nSharers)(true.B)))
  val weights = RegInit(VecInit(Seq.fill(nSharers)(0.U(log2Ceil(nSharers + 1).W))))

  // Make new group mask
  val groupMask = WireInit(VecInit(Seq.fill(nSharers)(0.U(group_num.W))))
  for (i <- 0 until nSharers) {
    when (!io.in(i).ldinput.idle && !(group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)).reduce(_||_))) {
      groupMask(i) := (1.U(group_num.W) << io.in(i).ldinput.group_id)
    }
  }
  val reducedMask = groupMask.reduce(_|_)

  // Make weight mask of each group
  val groupWeights = WireInit(VecInit(Seq.fill(group_num)(0.U(log2Ceil(nSharers + 1).W))))
  for (gId <- 0 until group_num) {
    val weightsForThisGroup = (0 until nSharers).map { sIdx =>
      val isMatch = !io.in(sIdx).ldinput.idle && io.in(sIdx).ldinput.group_id === gId.U
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

  for (i <- 0 until nSharers) {
    val theresMyGroup = group_data.map(gd => 
      gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)
    )
    val onGoing = group_data.zip(theresMyGroup).map{ case (gd, matched) => 
      matched && !gd.bits.mem_data(i).ldinput_completed
    }.reduce(_||_)

    val isWaiting = !io.in(i).ldinput.idle && !onGoing
    currentWaitingWire(i) := isWaiting

    when (isWaiting && !isWaitingReg(i)) {
      // new waiting
      waitMatrix(i) := isWaitingReg
    } .elsewhen (!isWaiting) {
      // waiting over
      waitMatrix(i) := 0.U
      for (j <- 0 until nSharers) {
        waitMatrix(j) := waitMatrix(j) & ~(1.U << i)
      }
    }
    val olderThanMeCount = PopCount(waitMatrix(i))
    when (isWaiting) {
      weights(i) := (nSharers.U - 1.U) - olderThanMeCount
    } .otherwise {
      weights(i) := 0.U
    }
  }

  isWaitingReg := currentWaitingWire.asUInt

  // Allocate groups to free slots
  val winnerMask = winnersVec.asUInt
  val allocMasks = Wire(Vec(group_num, UInt(group_num.W)))
  allocMasks(0) := winnerMask

  for (i <- 0 until group_num) {
    val canAllocate = !group_data(i).valid && allocMasks(i).orR
    val targetId = PriorityEncoder(allocMasks(i))
    
    val winnerGroupList = MuxCase(0.U, (0 until nSharers).map { sIdx =>
      val isMatch = !io.in(sIdx).ldinput.idle && io.in(sIdx).ldinput.group_id === targetId
      val isTopWeight = weights(sIdx) === groupWeights(targetId)
      (isMatch && isTopWeight) -> io.in(sIdx).ldinput.group_list
    })

    when (canAllocate) {
      group_data(i).valid := true.B
      group_data(i).bits.group_id := targetId
      
      val group_mask = VecInit(winnerGroupList.asBools)
      group_data(i).bits.mem_data.zip(group_mask).foreach { case (md, gm) =>
        md.ldinput_completed := !gm
        md.ex_completed := !gm
        md.st_completed := !gm
      }
    }
    
    if (i < group_num - 1) {
      allocMasks(i+1) := Mux(canAllocate, allocMasks(i) & ~(1.U(group_num.W) << targetId), allocMasks(i))
    }
  }

  // Set loop_full signals
  for (i <- 0 until nSharers) {
    val theresMyGroup = group_data.map(gd => 
      gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)
    )
    val onGoing = group_data.zip(theresMyGroup).map{ case (gd, matched) => 
      matched && !gd.bits.mem_data(i).ldinput_completed
    }.reduce(_||_)

    io.in(i).loop_full := !onGoing
  }

  for (i <- 0 until nSharers) {
    group_data.foreach { gd =>
      when (io.in(i).ldinput.idle && !ldinput_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)) {
        gd.bits.mem_data(i).ldinput_completed := true.B
      }

      when (io.in(i).ex.idle && !ex_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        gd.bits.mem_data(i).ex_completed := true.B
      }

      when (io.in(i).st.idle && !st_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).st.group_id)) {
        gd.bits.mem_data(i).st_completed := true.B
      }
    }
  }

  group_data.foreach { gd =>
    when (gd.valid) {
      gd.valid := !gd.bits.mem_data.map { md => md.ex_completed && md.st_completed }.reduce(_&&_)
    }
  }

  // if all members have completed, free the group
  group_data.foreach { gd =>
    when (gd.valid && gd.bits.mem_data.map { md => md.ex_completed && md.st_completed && md.ldinput_completed }.reduce(_&&_)) {
      gd.valid := false.B
      gd.bits.mem_data.foreach { md =>
        md.ex_completed := false.B
        md.st_completed := false.B
        md.ldinput_completed := false.B
      }
      gd.bits.group_id := 0.U
    }
  }

  for (i <- 0 until nSharers) {
    val group_list  = io.in(i).ex.group_list
    val group_mask  = VecInit(group_list.asBools)
    val lda_completed = WireInit(false.B)

    group_data.foreach { gd =>
      when (gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        lda_completed := gd.bits.mem_data.zip(group_mask).map { case (gdd, gmm) =>
          gdd.ldinput_completed
        }.reduce(_&&_)
      }
    }

    io.in(i).lda_ahead := lda_completed
  }

  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
  }

  // reset logic
  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
    group_data.foreach { gd =>
      gd.bits.mem_data.foreach(_.ldinput_completed := false.B)
      gd.bits.mem_data.foreach(_.ex_completed := false.B)
      gd.bits.mem_data.foreach(_.st_completed := false.B)
    }
    group_data.foreach(_.bits.group_id := 0.U)

    weights.foreach(_ := 0.U)
  }
}