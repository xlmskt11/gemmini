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
  large_iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
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
  val ldinput = Output(new LdInputState(group_w, large_iterator_bitwidth))
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

  val group_data = Reg(Vec(nSharers*concurrent_loops, Valid(new GroupData)))

  val ldinput_idle_delayed = RegNext(VecInit(io.in.map(_.ldinput.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))
  val ex_idle_delayed = RegNext(VecInit(io.in.map(_.ex.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))
  val st_idle_delayed = RegNext(VecInit(io.in.map(_.st.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))

  for (i <- 0 until nSharers) {
    io.in(i).loop_full := group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id) && gd.bits.mem_data(i).ldinput_completed && gd.bits.mem_data(i).st_completed && gd.bits.mem_data(i).ex_completed).reduce(_||_) || group_data.map(_.valid).reduce(_&&_)
  }

  val groupMask = WireInit(VecInit(Seq.fill(nSharers)(0.U(group_num.W))))
  for (i <- 0 until nSharers) {
    when (!io.in(i).ldinput.idle && !(group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)).reduce(_||_))) {
      groupMask(i) := (1.U(group_num.W) << io.in(i).ldinput.group_id)
    }
  }

  val alloc_id = MuxCase((nSharers*concurrent_loops - 1).U, group_data.zipWithIndex.map { case (e, i) => !e.valid -> i.U })
  val masks = Wire(Vec(nSharers+1, UInt(group_num.W)))
  masks(0) := groupMask.reduce(_|_)
  for (i <- 0 until nSharers) {
    val hasAny = masks(i).orR
    val idx = PriorityEncoder(masks(i))
    masks(i+1) := Mux(hasAny, masks(i) & ~(1.U(group_num.W) << idx), masks(i))

    when (hasAny) {
      group_data(alloc_id + i.U).valid := true.B
      group_data(alloc_id + i.U).bits.group_id := idx
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.ldinput_completed := false.B)
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.ex_completed := false.B)
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.st_completed := false.B)
    }
  }

  for (i <- 0 until nSharers) {
    group_data.foreach { gd =>
      when (io.in(i).ldinput.idle && !ldinput_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ldinput.group_id)) {
        gd.bits.mem_data(i).ldinput_completed := true.B
      }

      when (io.in(i).ex.idle && !ex_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        gd.bits.mem_data(i).ex_completed := true.B

        val group_list = io.in(i).ex.group_list
        val group_mask  = VecInit(group_list.asBools)
        gd.bits.mem_data.zip(group_mask).foreach { case (md, gm) =>
          when (!gm) {
            md.ex_completed := true.B
            md.st_completed := true.B
          }
        }

        // val all_completed = gd.bits.mem_data.zip(group_mask).zipWithIndex.map{ case ((md, gm), k) => if (k == i) true.B else md.ex_completed === gm}.reduce(_&&_)
        // val completed_list = gd.bits.mem_data.zip(group_mask).map { case (md, gm) => gm === md.ex_completed }
        // val all_completed = completed_list.zip(io.in).map { case (cl, in) => cl || (in.ex.idle && gd.bits.group_id === in.ex.group_id)}.reduce(_&&_)
        // when (all_completed) {
        //   gd.valid := false.B
        // }
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

  for (i <- 0 until nSharers) {
    val group_list  = io.in(i).ex.group_list
    val group_mask  = VecInit(group_list.asBools)
    val lda_completed = WireInit(false.B)

    group_data.foreach { gd =>
      when (gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        lda_completed := gd.bits.mem_data.zip(group_mask).map { case (gdd, gmm) =>
          !gmm || gdd.ldinput_completed
        }.reduce(_&&_)
      }
    }

    io.in(i).lda_ahead := lda_completed
  }

  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
  }
}