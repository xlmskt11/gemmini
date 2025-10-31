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
  iterator_bitwidth: Int
) extends Bundle {
  val group_id      = UInt(group_w.W)
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
  val ldb = Output(new LdBState(group_w, iterator_bitwidth))
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
  val group_w = log2Up(group_num) + 1

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

  val group_data = Reg(Vec(nSharers*concurrent_loops, Valid(new GroupData)))
  val ldb_idle_delayed = RegNext(VecInit(io.in.map(_.ldb.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))
  val ex_idle_delayed = RegNext(VecInit(io.in.map(_.ex.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))
  val stc_idle_delayed = RegNext(VecInit(io.in.map(_.stc.idle)), 1.U.asTypeOf(Vec(nSharers, Bool())))

  for (i <- 0 until nSharers) {
    io.in(i).loop_full := group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id) && gd.bits.mem_data(i).stc_completed && gd.bits.mem_data(i).ex_completed).reduce(_||_) || group_data.map(_.valid).reduce(_&&_)
  }

  val groupMask = WireInit(VecInit(Seq.fill(nSharers)(0.U(group_num.W))))
  for (i <- 0 until nSharers) {
    when (!io.in(i).ldb.idle && !(group_data.map( gd => gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)).reduce(_||_))) {
      groupMask(i) := (1.U(group_num.W) << io.in(i).ldb.group_id)
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
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.ldb_end_data.valid := false.B)
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.ex_completed := false.B)
      group_data(alloc_id + i.U).bits.mem_data.foreach(_.stc_completed := false.B)
    }
  }

  for (i <- 0 until nSharers) {
    group_data.foreach { gd =>
      when (io.in(i).ldb.idle && !ldb_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ldb.group_id)) {
        gd.bits.mem_data(i).ldb_end_data.valid := true.B
        gd.bits.mem_data(i).ldb_end_data.bits.max_k := io.in(i).ldb.max_k
        gd.bits.mem_data(i).ldb_end_data.bits.k_offset := io.in(i).ldb.k_offset
      }

      when (io.in(i).ex.idle && !ex_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).ex.group_id)) {
        gd.bits.mem_data(i).ex_completed := true.B

        val group_list = io.in(i).ex.group_list
        val group_mask  = VecInit(group_list.asBools)
        gd.bits.mem_data.zip(group_mask).foreach { case (md, gm) =>
          when (!gm) {
            md.ex_completed := true.B
            md.stc_completed := true.B
          }
        }

        // val all_completed = gd.bits.mem_data.zip(group_mask).zipWithIndex.map{ case ((md, gm), k) => if (k == i) true.B else md.ex_completed === gm}.reduce(_&&_)
        // val completed_list = gd.bits.mem_data.zip(group_mask).map { case (md, gm) => gm === md.ex_completed }
        // val all_completed = completed_list.zip(io.in).map { case (cl, in) => cl || (in.ex.idle && gd.bits.group_id === in.ex.group_id)}.reduce(_&&_)
        // when (all_completed) {
        //   gd.valid := false.B
        // }
      }

      when (io.in(i).stc.idle && !stc_idle_delayed(i) && gd.valid && (gd.bits.group_id === io.in(i).stc.group_id)) {
        gd.bits.mem_data(i).stc_completed := true.B
      }
    }
  }

  group_data.foreach { gd =>
    when (gd.valid) {
      gd.valid := !gd.bits.mem_data.map { md => md.ex_completed && md.stc_completed }.reduce(_&&_)
    }
  }

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

    val ld_ahead = io.in.zip(group_mask).map{ case (in, m) => (io.in(i).ex.group_id === in.ldb.group_id) && m && (io.in(i).ex.k >= in.ldb.k_offset) && ((in.ldb.k_offset + in.ldb.k > io.in(i).ex.k) || ((in.ldb.k_offset + in.ldb.k === io.in(i).ex.k && in.ldb.j > io.in(i).ex.j)))}.reduce(_||_)

    io.in(i).ldb_ahead := ldb_completed || ld_ahead
  }

  when (reset.asBool) {
    group_data.foreach(_.valid := false.B)
  }
}