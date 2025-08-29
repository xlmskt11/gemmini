package gemmini

import chisel3._
import chisel3.util._
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

// made
class ExtMemRWIO_4 extends Bundle {
  val read_en = Bool()
  val read_addr = UInt()

  val write_en = Bool()
  val write_addr = UInt()
  val write_data = UInt()
  val write_mask = UInt()
}

// made
class ExtMemReadIO extends Bundle {
  val read_addr = UInt()
}

// made
class ExtMemWriteIO extends Bundle {
  val write_addr = UInt()
  val write_data = UInt()
  val write_mask = UInt()
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

// made
private object ArbiterCtrl {
  def apply(request: Seq[Bool]): Seq[Bool] = request.length match {
    case 0 => Seq()
    case 1 => Seq(true.B)
    case _ => true.B +: request.tail.init.scanLeft(request.head)(_ || _).map(!_)
  }
}

// made
class RrArbiter[T <: Data](val gen: T, val n: Int) extends Module {
  val io = IO(new ArbiterIO(gen, n))

  val lastGrant = RegInit((n - 1).U(log2Ceil(n).W))

  io.chosen := (n - 1).asUInt
  io.out.bits := io.in(n - 1).bits
  io.out.valid := false.B

  for (i <- n - 1 to 0 by -1) {
    val idx = (lastGrant + 1.U + i.U)(log2Ceil(n) - 1, 0)
    when(io.in(idx).valid) {
      io.chosen := idx
      io.out.bits := io.in(idx).bits
      io.out.valid := true.B
    }
  }

  for ((in, idx) <- io.in.zipWithIndex) {
    in.ready := (io.out.valid && (io.chosen === idx.U) && io.out.ready)
  }

  when(io.out.fire) {
    lastGrant := io.chosen
  }
}

// made
class SharedSyncReadMem_4(nSharers: Int, depth: Int, mask_len: Int, data_len: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
  })

  val mem = SRAM(depth, Vec(mask_len, UInt(data_len.W)), 0, 0, 4)

  for (i <- 0 until nSharers) {
    mem.readwritePorts(i).enable := false.B
    when(io.in(i).write_en || io.in(i).read_en) {
      mem.readwritePorts(i).enable := true.B
    }

    mem.readwritePorts(i).address := io.in(i).read_addr
    mem.readwritePorts(i).isWrite := false.B
    when(io.in(i).write_en) {
      mem.readwritePorts(i).address := io.in(i).write_addr
      mem.readwritePorts(i).isWrite := true.B
    }

    mem.readwritePorts(i).writeData := io.in(i).write_data.asTypeOf(Vec(mask_len, UInt(data_len.W)))
    val rdata = WireInit(Vec(mask_len, UInt(data_len.W)), mem.readwritePorts(i).readData)
    io.in(i).read_data := rdata.asUInt
    
    mem.readwritePorts(i).mask.foreach { m =>
      m := io.in(i).write_mask.asTypeOf(Vec(mask_len, Bool()))
    }
  }
}

// class SharedSyncReadMem_4(nSharers: Int, depth: Int, mask_len: Int, data_len: Int) extends Module {
//   val io = IO(new Bundle {
//     val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
//   })

//   val mem = SyncReadMem(depth, Vec(mask_len, UInt(data_len.W)))

//   val q = Seq.fill(nSharers)(Module(new Queue(new ExtMemRWIO_4(), 1, true, true)))

//   for (i <- 0 until nSharers) {
//     q(i).io.enq.valid := io.in(i).read_en || io.in(i).write_en

//     q(i).io.enq.bits.read_en := io.in(i).read_en
//     q(i).io.enq.bits.read_addr := io.in(i).read_addr

//     q(i).io.enq.bits.write_en := io.in(i).write_en
//     q(i).io.enq.bits.write_addr := io.in(i).write_addr
//     q(i).io.enq.bits.write_data := io.in(i).write_data
//     q(i).io.enq.bits.write_mask := io.in(i).write_mask

//     io.in(i).rw_ready := q(i).io.enq.ready
//   }

//   val arb = Module(new RRArbiter(new ExtMemRWIO_4(), nSharers))

//   // io.in.foreach(_.read_data := 0.U)
//   io.in.foreach(_.read_valid := false.B)
//   //io.in.foreach(_.write_over := false.B)

//   arb.io.out.ready := true.B

//   for (i <- 0 until nSharers) {
//     arb.io.in(i) <> q(i).io.deq
//   }
  
//   val en = arb.io.out.valid

//   val ren = arb.io.out.bits.read_en
//   val raddr = arb.io.out.bits.read_addr

//   val wen = arb.io.out.bits.write_en
//   val waddr = arb.io.out.bits.write_addr
//   val wdata = arb.io.out.bits.write_data
//   val wmask = arb.io.out.bits.write_mask

//   val rdata = mem.read(raddr, en && ren && !wen)
//   io.in.foreach(_.read_data := rdata.asUInt)

//   when (en) {
//     when (ren && !wen) {
//       io.in(arb.io.chosen).read_valid := true.B
//     }

//     when (wen) {
//       mem.write(waddr, wdata.asTypeOf(Vec(mask_len, UInt(data_len.W))), wmask.asTypeOf(Vec(mask_len, Bool())))
//     }
//   }
// }

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
class SharedExtMem_4(
  sp_banks: Int, acc_banks: Int, acc_sub_banks: Int,
  sp_depth: Int, sp_mask_len: Int, sp_data_len: Int,
  acc_depth: Int, acc_mask_len: Int, acc_data_len: Int
) extends Module {
  val nSharers = 4
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtSpadMemIO_4(sp_banks, acc_banks, acc_sub_banks)))
  })
  for (i <- 0 until sp_banks) {
    val spad_mem = Module(new SharedSyncReadMem_4(nSharers, sp_depth, sp_mask_len, sp_data_len))
    for (w <- 0 until nSharers) {
      spad_mem.io.in(w) <> io.in(w).spad(i)
    }
  }
  for (i <- 0 until acc_banks) {
    for (s <- 0 until acc_sub_banks) {
      val acc_mem = Module(new SharedSyncReadMem_4(nSharers, acc_depth, acc_mask_len, acc_data_len))
      for (w <- 0 until nSharers) {
        acc_mem.io.in(w) <> io.in(w).acc(i)(s)
      }
    }
  }
}
