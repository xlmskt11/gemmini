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
class SharedSyncReadMem_4(nSharers: Int, depth: Int, mask_len: Int, data_len: Int) extends Module {
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
  })

  // val mem = SRAM(depth, Vec(mask_len, UInt(data_len.W)), 0, 0, 4, true)
  val mem = SRAM.masked(depth, Vec(mask_len, UInt(data_len.W)), 4, 4, 0)

  for (i <- 0 until nSharers) {
    mem.readPorts(i).enable := false.B
    mem.writePorts(i).enable := false.B
    when(io.in(i).read_en) {
      mem.readPorts(i).enable := true.B
    }.elsewhen(io.in(i).write_en) {
      mem.writePorts(i).enable := true.B
    }

    mem.readPorts(i).address := io.in(i).read_addr
    mem.writePorts(i).address := io.in(i).write_addr

    mem.writePorts(i).data := io.in(i).write_data.asTypeOf(Vec(mask_len, UInt(data_len.W)))
    val rdata = WireInit(Vec(mask_len, UInt(data_len.W)), mem.readPorts(i).data)
    io.in(i).read_data := rdata.asUInt
    
    mem.writePorts(i).mask.foreach { m =>
      m := io.in(i).write_mask.asTypeOf(Vec(mask_len, Bool()))
    }
  }
}

/*
 * LVT-based 4R4W memory using SyncReadMem.
 */
class SharedSyncReadMem_4_LVT(
  nSharers: Int,
  depth: Int,
  mask_len: Int,
  data_len: Int
) extends Module {
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtMemIO_4()))
  })

  val W = nSharers
  val R = nSharers
  val addrWidth = log2Ceil(depth)
  val writerBits = log2Ceil(W)

  // 1. Data Replicas (BRAM 기반)
  val mem = Seq.fill(W, R) {
    SyncReadMem(depth, Vec(mask_len, UInt(data_len.W)))
  }

  // 2. LVT (LUTRAM 기반 선호)
  // Reg 대신 Mem을 사용하여 자원 절약. asyncRead가 가능한 Mem 사용 시 sel 타이밍 유지 가능
  val lvt = Mem(depth, UInt(writerBits.W))

  // --- Write Logic ---
  for (i <- 0 until W) {
    val wPort = io.in(i)
    when(wPort.write_en) {
      lvt.write(wPort.write_addr, i.U) // 누가 썼는지 기록
      for (r <- 0 until R) {
        mem(i)(r).write(wPort.write_addr, 
                             wPort.write_data.asTypeOf(Vec(mask_len, UInt(data_len.W))), 
                             wPort.write_mask.asTypeOf(Vec(mask_len, Bool())))
      }
    }
  }

  // --- Read Logic ---
  for (r <- 0 until R) {
    val rPort = io.in(r)
    // 읽기 활성화 시 현재 주소의 최신 작성자 index 확인
    // Mem의 read(addr)은 조합회로(Async)처럼 동작하거나 동기식으로 설정 가능
    val sel = lvt.read(rPort.read_addr) 
    val sel_p1 = RegNext(sel)
    
    val cand = Wire(Vec(W, Vec(mask_len, UInt(data_len.W))))
    for (w <- 0 until W) {
      // SyncReadMem은 t+1 시점에 데이터를 출력
      cand(w) := mem(w)(r).read(rPort.read_addr, rPort.read_en)
    }

    // t+1 시점에 최신 작성자 데이터를 선택
    val chosen = cand(sel_p1)
    rPort.read_data := chosen.asUInt
  }
}

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
  nSharers: Int,
  sp_banks: Int, acc_banks: Int, acc_sub_banks: Int,
  sp_depth: Int, sp_mask_len: Int, sp_data_len: Int,
  acc_depth: Int, acc_mask_len: Int, acc_data_len: Int
) extends Module {
  val io = IO(new Bundle {
    val in = Vec(nSharers, Flipped(new ExtSpadMemIO_4(sp_banks, acc_banks, acc_sub_banks)))
  })
  for (i <- 0 until sp_banks) {
    val spad_mem = Module(new SharedSyncReadMem_4_LVT(nSharers, sp_depth, sp_mask_len, sp_data_len))
    for (w <- 0 until nSharers) {
      spad_mem.io.in(w) <> io.in(w).spad(i)
    }
  }
  for (i <- 0 until acc_banks) {
    for (s <- 0 until acc_sub_banks) {
      val acc_mem = Module(new SharedSyncReadMem_4_LVT(nSharers, acc_depth, acc_mask_len, acc_data_len))
      for (w <- 0 until nSharers) {
        acc_mem.io.in(w) <> io.in(w).acc(i)(s)
      }
    }
  }
}
