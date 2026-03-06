
package gemmini

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy.{LazyModule, LazyModuleImp}
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._

import Util._

class ScratchpadMemReadRequest[U <: Data](local_addr_t: LocalAddr, scale_t_bits: Int)(implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val laddr = local_addr_t.cloneType

  val cols = UInt(16.W) // TODO don't use a magic number for the width here
  val repeats = UInt(16.W) // TODO don't use a magic number for the width here
  val scale = UInt(scale_t_bits.W)
  val has_acc_bitwidth = Bool()
  val all_zeros = Bool()
  val block_stride = UInt(16.W) // TODO magic numbers
  val pixel_repeats = UInt(8.W) // TODO magic numbers
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
  val status = new MStatus

}

class ScratchpadMemWriteRequest(local_addr_t: LocalAddr, acc_t_bits: Int, scale_t_bits: Int)
                              (implicit p: Parameters) extends CoreBundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val laddr = local_addr_t.cloneType

  val acc_act = UInt(Activation.bitwidth.W) // TODO don't use a magic number for the width here
  val acc_scale = UInt(scale_t_bits.W)
  val acc_igelu_qb = UInt(acc_t_bits.W)
  val acc_igelu_qc = UInt(acc_t_bits.W)
  val acc_iexp_qln2 = UInt(acc_t_bits.W)
  val acc_iexp_qln2_inv = UInt(acc_t_bits.W)
  val acc_norm_stats_id = UInt(8.W) // TODO magic number

  val len = UInt(16.W) // TODO don't use a magic number for the width here
  val block = UInt(8.W) // TODO don't use a magic number for the width here

  val cmd_id = UInt(8.W) // TODO don't use a magic number here
  val status = new MStatus

  // Pooling variables
  val pool_en = Bool()
  val store_en = Bool()

}

class ScratchpadMemWriteResponse extends Bundle {
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
}

class ScratchpadMemReadResponse extends Bundle {
  val bytesRead = UInt(16.W) // TODO magic number here
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
}

class ScratchpadReadMemIO[U <: Data](local_addr_t: LocalAddr, scale_t_bits: Int)(implicit p: Parameters) extends CoreBundle {
  val req = Decoupled(new ScratchpadMemReadRequest(local_addr_t, scale_t_bits))
  val resp = Flipped(Valid(new ScratchpadMemReadResponse))
}

class ScratchpadWriteMemIO(local_addr_t: LocalAddr, acc_t_bits: Int, scale_t_bits: Int)
                         (implicit p: Parameters) extends CoreBundle {
  val req = Decoupled(new ScratchpadMemWriteRequest(local_addr_t, acc_t_bits, scale_t_bits))
  val resp = Flipped(Valid(new ScratchpadMemWriteResponse))
}

class ScratchpadReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class ScratchpadReadResp(val w: Int) extends Bundle {
  val data = UInt(w.W)
  val fromDMA = Bool()
}

class ScratchpadReadIO(val n: Int, val w: Int) extends Bundle {
  val req = Decoupled(new ScratchpadReadReq(n))
  val resp = Flipped(Decoupled(new ScratchpadReadResp(w)))
}

class ScratchpadWriteIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val en = Output(Bool())
  val addr = Output(UInt(log2Ceil(n).W))
  val mask = Output(Vec(mask_len, Bool()))
  val data = Output(UInt(w.W))
}

class ExtSpadSubBankAdapter(n: Int, subBanks: Int, w: Int, mask_len: Int, capacity: Int) extends Module {
  require(subBanks > 0 && isPow2(subBanks))
  require(n % subBanks == 0)

  private val subBits = log2Ceil(subBanks)
  private val selWidth = (1 max subBits)

  def subIdx(addr: UInt): UInt = if (subBanks == 1) 0.U(selWidth.W) else addr(subBits - 1, 0)
  def subAddr(addr: UInt): UInt = if (subBanks == 1) addr else (addr >> subBits)

  val io = IO(new Bundle {
    val bank = new Bundle {
      val read = Flipped(new ScratchpadReadIO(n, w))
      val write = Flipped(Decoupled(new ExtScratchpadWriteReq(n, w, mask_len)))
      val grant = Flipped(Decoupled(new BankExWriteGrantReq(n)))
      val remind = Flipped(Decoupled(new BankExWriteRemindReq(n)))
    }
    val ext = Vec(subBanks, new ExtScratchpadBankIO(n / subBanks, w, mask_len, capacity))
  })

  val readSelQ = Module(new Queue(UInt(selWidth.W), capacity, pipe = true, flow = false))

  io.bank.read.req.ready := false.B
  io.bank.read.resp.valid := false.B
  io.bank.read.resp.bits := DontCare
  io.bank.write.ready := false.B
  io.bank.grant.ready := false.B
  io.bank.remind.ready := false.B
  readSelQ.io.enq.valid := false.B
  readSelQ.io.enq.bits := 0.U
  readSelQ.io.deq.ready := false.B

  for (s <- 0 until subBanks) {
    io.ext(s).read.req.valid := false.B
    io.ext(s).read.req.bits := DontCare
    io.ext(s).read.resp.ready := false.B

    io.ext(s).write.valid := false.B
    io.ext(s).write.bits := DontCare
    io.ext(s).write.bits.exwrite := false.B
    io.ext(s).grant.valid := false.B
    io.ext(s).grant.bits := false.B
    io.ext(s).remind.valid := false.B
    io.ext(s).remind.bits := false.B
  }

  val readSel = subIdx(io.bank.read.req.bits.addr)
  val readSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(readSel, subBanks)
  val selectedReadReady = Mux1H(readSelOH.asBools.zip(io.ext.map(_.read.req.ready)))

  io.bank.read.req.ready := selectedReadReady && readSelQ.io.enq.ready
  readSelQ.io.enq.valid := io.bank.read.req.valid && selectedReadReady
  readSelQ.io.enq.bits := readSel

  for (s <- 0 until subBanks) {
    when (readSelOH(s)) {
      io.ext(s).read.req.valid := io.bank.read.req.valid && readSelQ.io.enq.ready
      io.ext(s).read.req.bits.addr := subAddr(io.bank.read.req.bits.addr)
      io.ext(s).read.req.bits.fromDMA := io.bank.read.req.bits.fromDMA
    }
  }

  val headSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(readSelQ.io.deq.bits, subBanks)
  val selectedRespValid = Mux1H(headSelOH.asBools.zip(io.ext.map(_.read.resp.valid)))
  val selectedRespBits = Mux1H(headSelOH.asBools.zip(io.ext.map(_.read.resp.bits)))

  io.bank.read.resp.valid := readSelQ.io.deq.valid && selectedRespValid
  when (readSelQ.io.deq.valid) {
    io.bank.read.resp.bits := selectedRespBits
  }

  readSelQ.io.deq.ready := io.bank.read.resp.fire
  for (s <- 0 until subBanks) {
    io.ext(s).read.resp.ready := readSelQ.io.deq.valid && headSelOH(s) && io.bank.read.resp.ready
  }

  val writeSel = subIdx(io.bank.write.bits.addr)
  val writeSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(writeSel, subBanks)
  val selectedWriteReady = Mux1H(writeSelOH.asBools.zip(io.ext.map(_.write.ready)))
  io.bank.write.ready := selectedWriteReady

  for (s <- 0 until subBanks) {
    when (writeSelOH(s)) {
      io.ext(s).write.valid := io.bank.write.valid
      io.ext(s).write.bits.addr := subAddr(io.bank.write.bits.addr)
      io.ext(s).write.bits.mask := io.bank.write.bits.mask
      io.ext(s).write.bits.data := io.bank.write.bits.data
      io.ext(s).write.bits.exwrite := io.bank.write.bits.exwrite
    }
  }

  val grantSel = subIdx(io.bank.grant.bits.addr)
  val grantSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(grantSel, subBanks)
  val selectedGrantReady = Mux1H(grantSelOH.asBools.zip(io.ext.map(_.grant.ready)))
  io.bank.grant.ready := selectedGrantReady
  for (s <- 0 until subBanks) {
    when (grantSelOH(s)) {
      io.ext(s).grant.valid := io.bank.grant.valid
      io.ext(s).grant.bits := true.B
    }
  }

  val remindSel = subIdx(io.bank.remind.bits.addr)
  val remindSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(remindSel, subBanks)
  val selectedRemindReady = Mux1H(remindSelOH.asBools.zip(io.ext.map(_.remind.ready)))
  io.bank.remind.ready := selectedRemindReady
  for (s <- 0 until subBanks) {
    when (remindSelOH(s)) {
      io.ext(s).remind.valid := io.bank.remind.valid
      io.ext(s).remind.bits := true.B
    }
  }
}

class ExtAccSubBankAdapter[T <: Data: Arithmetic, U <: Data](
  n: Int, t: Vec[Vec[T]], scale_t: U, subBanks: Int, capacity: Int
) extends Module {
  require(subBanks > 0 && isPow2(subBanks))
  require(n % subBanks == 0)

  private val subBits = log2Ceil(subBanks)
  private val selWidth = (1 max subBits)

  def subIdx(addr: UInt): UInt = if (subBanks == 1) 0.U(selWidth.W) else addr(subBits - 1, 0)
  def subAddr(addr: UInt): UInt = if (subBanks == 1) addr else (addr >> subBits)

  class AccReadRespMeta extends Bundle {
    val fromDMA = Bool()
    val scale = scale_t.cloneType
    val igelu_qb = t.head.head.cloneType
    val igelu_qc = t.head.head.cloneType
    val iexp_qln2 = t.head.head.cloneType
    val iexp_qln2_inv = t.head.head.cloneType
    val act = UInt(Activation.bitwidth.W)
  }

  val io = IO(new Bundle {
    val bank = new Bundle {
      val read = Flipped(new AccumulatorReadIO(n, t, scale_t))
      val write = Flipped(Decoupled(new ExtAccumulatorWriteReq(n, t)))
      val grant = Flipped(Decoupled(new BankExWriteGrantReq(n)))
      val remind = Flipped(Decoupled(new BankExWriteRemindReq(n)))
    }
    val ext = Vec(subBanks, new ExtAccumulatorBankIO(n / subBanks, t, capacity))
  })

  val readSelQ = Module(new Queue(UInt(selWidth.W), capacity, pipe = true, flow = false))
  val readMetaQ = Module(new Queue(new AccReadRespMeta, capacity, pipe = true, flow = false))

  io.bank.read.req.ready := false.B
  io.bank.read.resp.valid := false.B
  io.bank.read.resp.bits := DontCare
  io.bank.write.ready := false.B
  io.bank.grant.ready := false.B
  io.bank.remind.ready := false.B

  readSelQ.io.enq.valid := false.B
  readSelQ.io.enq.bits := 0.U
  readSelQ.io.deq.ready := false.B
  readMetaQ.io.enq.valid := false.B
  readMetaQ.io.enq.bits := DontCare
  readMetaQ.io.deq.ready := false.B

  for (s <- 0 until subBanks) {
    io.ext(s).read.req.valid := false.B
    io.ext(s).read.req.bits := DontCare
    io.ext(s).read.req.bits.idx := 0.U
    io.ext(s).read.resp.ready := false.B

    io.ext(s).write.valid := false.B
    io.ext(s).write.bits := DontCare
    io.ext(s).write.bits.exwrite := false.B
    io.ext(s).grant.valid := false.B
    io.ext(s).grant.bits := false.B
    io.ext(s).remind.valid := false.B
    io.ext(s).remind.bits := false.B
  }

  val readSel = subIdx(io.bank.read.req.bits.addr)
  val readSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(readSel, subBanks)
  val selectedReadReady = Mux1H(readSelOH.asBools.zip(io.ext.map(_.read.req.ready)))

  io.bank.read.req.ready := selectedReadReady && readSelQ.io.enq.ready && readMetaQ.io.enq.ready

  readSelQ.io.enq.valid := io.bank.read.req.valid && selectedReadReady
  readSelQ.io.enq.bits := readSel
  readMetaQ.io.enq.valid := io.bank.read.req.valid && selectedReadReady
  readMetaQ.io.enq.bits.fromDMA := io.bank.read.req.bits.fromDMA
  readMetaQ.io.enq.bits.scale := io.bank.read.req.bits.scale
  readMetaQ.io.enq.bits.igelu_qb := io.bank.read.req.bits.igelu_qb
  readMetaQ.io.enq.bits.igelu_qc := io.bank.read.req.bits.igelu_qc
  readMetaQ.io.enq.bits.iexp_qln2 := io.bank.read.req.bits.iexp_qln2
  readMetaQ.io.enq.bits.iexp_qln2_inv := io.bank.read.req.bits.iexp_qln2_inv
  readMetaQ.io.enq.bits.act := io.bank.read.req.bits.act

  for (s <- 0 until subBanks) {
    when (readSelOH(s)) {
      io.ext(s).read.req.valid := io.bank.read.req.valid && readSelQ.io.enq.ready && readMetaQ.io.enq.ready
      io.ext(s).read.req.bits.addr := subAddr(io.bank.read.req.bits.addr)
      io.ext(s).read.req.bits.full := io.bank.read.req.bits.full
      io.ext(s).read.req.bits.idx := 0.U
    }
  }

  val headSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(readSelQ.io.deq.bits, subBanks)
  val selectedRespValid = Mux1H(headSelOH.asBools.zip(io.ext.map(_.read.resp.valid)))
  val selectedRespBits = Mux1H(headSelOH.asBools.zip(io.ext.map(_.read.resp.bits)))

  io.bank.read.resp.valid := readSelQ.io.deq.valid && readMetaQ.io.deq.valid && selectedRespValid
  when (readSelQ.io.deq.valid && readMetaQ.io.deq.valid) {
    io.bank.read.resp.bits.data := selectedRespBits.data
    io.bank.read.resp.bits.fromDMA := readMetaQ.io.deq.bits.fromDMA
    io.bank.read.resp.bits.scale := readMetaQ.io.deq.bits.scale
    io.bank.read.resp.bits.igelu_qb := readMetaQ.io.deq.bits.igelu_qb
    io.bank.read.resp.bits.igelu_qc := readMetaQ.io.deq.bits.igelu_qc
    io.bank.read.resp.bits.iexp_qln2 := readMetaQ.io.deq.bits.iexp_qln2
    io.bank.read.resp.bits.iexp_qln2_inv := readMetaQ.io.deq.bits.iexp_qln2_inv
    io.bank.read.resp.bits.act := readMetaQ.io.deq.bits.act
    io.bank.read.resp.bits.acc_bank_id := 0.U
  }

  readSelQ.io.deq.ready := io.bank.read.resp.fire
  readMetaQ.io.deq.ready := io.bank.read.resp.fire
  for (s <- 0 until subBanks) {
    io.ext(s).read.resp.ready := readSelQ.io.deq.valid && readMetaQ.io.deq.valid && headSelOH(s) && io.bank.read.resp.ready
  }

  val writeSel = subIdx(io.bank.write.bits.addr)
  val writeSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(writeSel, subBanks)
  val selectedWriteReady = Mux1H(writeSelOH.asBools.zip(io.ext.map(_.write.ready)))

  io.bank.write.ready := selectedWriteReady

  for (s <- 0 until subBanks) {
    when (writeSelOH(s)) {
      io.ext(s).write.valid := io.bank.write.valid
      io.ext(s).write.bits.addr := subAddr(io.bank.write.bits.addr)
      io.ext(s).write.bits.data := io.bank.write.bits.data
      io.ext(s).write.bits.acc := io.bank.write.bits.acc
      io.ext(s).write.bits.mask := io.bank.write.bits.mask
      io.ext(s).write.bits.exwrite := io.bank.write.bits.exwrite
    }
  }

  val grantSel = subIdx(io.bank.grant.bits.addr)
  val grantSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(grantSel, subBanks)
  val selectedGrantReady = Mux1H(grantSelOH.asBools.zip(io.ext.map(_.grant.ready)))
  io.bank.grant.ready := selectedGrantReady
  for (s <- 0 until subBanks) {
    when (grantSelOH(s)) {
      io.ext(s).grant.valid := io.bank.grant.valid
      io.ext(s).grant.bits := true.B
    }
  }

  val remindSel = subIdx(io.bank.remind.bits.addr)
  val remindSelOH = if (subBanks == 1) 1.U(1.W) else UIntToOH(remindSel, subBanks)
  val selectedRemindReady = Mux1H(remindSelOH.asBools.zip(io.ext.map(_.remind.ready)))
  io.bank.remind.ready := selectedRemindReady
  for (s <- 0 until subBanks) {
    when (remindSelOH(s)) {
      io.ext(s).remind.valid := io.bank.remind.valid
      io.ext(s).remind.bits := true.B
    }
  }
}

class ScratchpadBank(n: Int, w: Int, aligned_to: Int, single_ported: Boolean, use_shared_ext_mem: Boolean, is_dummy: Boolean) extends Module {
  // This is essentially a pipelined SRAM with the ability to stall pipeline stages

  require(w % aligned_to == 0 || w < aligned_to)
  val mask_len = (w / (aligned_to * 8)) max 1 // How many mask bits are there?
  val mask_elem = UInt((w min (aligned_to * 8)).W) // What datatype does each mask bit correspond to?

  val io = IO(new Bundle {
    val read = Flipped(new ScratchpadReadIO(n, w))
    // changed
    val write = Flipped(new ScratchpadWriteIO(n, w, mask_len))
    // val write = Flipped(Decoupled(new ScratchpadWriteIO(n, w, mask_len)))
    // changed
    // val ext_mem = if (use_shared_ext_mem) Some(new ExtMemIO) else None
    val ext_mem = if (use_shared_ext_mem) Some(new ExtMemIO_4) else None
  })

  val (read, write) = if (is_dummy) {
    // changed
    def read(addr: UInt, ren: Bool): Data = 0.U
    // def read(addr: UInt, ren: Bool): (Data, Bool) = (0.U, false.B)
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]): Unit = { }
    (read _, write _)
  } else if (use_shared_ext_mem) {
    // changed
    def read(addr: UInt, ren: Bool): Data = {
      io.ext_mem.get.read_en := ren
      io.ext_mem.get.read_addr := addr
      io.ext_mem.get.read_data
    }
    // def read(addr: UInt, ren: Bool): (Data, Bool) = {
    //   io.ext_mem.get.read_en := ren
    //   io.ext_mem.get.read_addr := addr
    //   (io.ext_mem.get.read_data, io.ext_mem.get.read_valid)
    // }
    io.ext_mem.get.write_en := false.B
    io.ext_mem.get.write_addr := DontCare
    io.ext_mem.get.write_data := DontCare
    io.ext_mem.get.write_mask := DontCare
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]) = {
      io.ext_mem.get.write_en := true.B
      io.ext_mem.get.write_addr := addr
      io.ext_mem.get.write_data := wdata.asUInt
      io.ext_mem.get.write_mask := wmask.asUInt
    }
    (read _, write _)
  } else {
    val mem = SyncReadMem(n, Vec(mask_len, mask_elem))
    // changed
    def read(addr: UInt, ren: Bool): Data = mem.read(addr, ren)
    // def read(addr: UInt, ren: Bool): (Data, Bool) = (mem.read(addr, ren), ren)
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]) = mem.write(addr, wdata, wmask)
    (read _, write _)
  }

  // When the scratchpad is single-ported, the writes take precedence
  // changed
  val singleport_busy_with_write = single_ported.B && io.write.en
  // val singleport_busy_with_write = single_ported.B && io.write.bits.en && io.write.fire

  // changed
  when (io.write.en) {
    if (aligned_to >= w)
      write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
    else
      write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), io.write.mask)
  }
  // val wen = io.write.bits.en && io.write.fire
  // val write_over = WireInit(false.B)
  // when (wen) {
  //   val write_overd = if (aligned_to >= w)
  //     write(io.write.bits.addr, io.write.bits.data.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
  //   else
  //     write(io.write.bits.addr, io.write.bits.data.asTypeOf(Vec(mask_len, mask_elem)), io.write.bits.mask)
  //   write_over := write_overd
  // }

  val raddr = io.read.req.bits.addr
  val ren = io.read.req.fire
  // changed
  val rdata = if (single_ported) {
    assert(!(ren && io.write.en))
    read(raddr, ren && !io.write.en).asUInt
  } else {
    read(raddr, ren).asUInt
  }
  // val (rdatad, rvalid) = if (single_ported) {
  //   assert(!(ren && io.write.en))
  //   read(raddr, ren && !io.write.en)
  // } else {
  //   read(raddr, ren)
  // }
  // val rdata = rdatad.asUInt
  // changed end

  val fromDMA = io.read.req.bits.fromDMA

  // Make a queue which buffers the result of an SRAM read if it can't immediately be consumed
  val q = Module(new Queue(new ScratchpadReadResp(w), 1, true, true))

  // changed
  q.io.enq.valid := RegNext(ren)
  // q.io.enq.valid := RegNext(rvalid)
  q.io.enq.bits.data := rdata
  q.io.enq.bits.fromDMA := RegNext(fromDMA)

  val q_will_be_empty = (q.io.count +& q.io.enq.fire) - q.io.deq.fire === 0.U
  // changed
  io.read.req.ready := q_will_be_empty && !singleport_busy_with_write
  // io.read.req.ready := q_will_be_empty && !singleport_busy_with_write && RegNext(io.ext_mem.get.rw_ready)

  // made
  // io.write.ready := !proceeding

  io.read.resp <> q.io.deq
}


class Scratchpad[T <: Data, U <: Data, V <: Data](config: GemminiArrayConfig[T, U, V])
    (implicit p: Parameters, ev: Arithmetic[T]) extends LazyModule {

  import config._
  import ev._

  val maxBytes = dma_maxbytes
  val dataBits = dma_buswidth

  val block_rows = meshRows * tileRows
  val block_cols = meshColumns * tileColumns
  val spad_w = inputType.getWidth *  block_cols
  val acc_w = accType.getWidth * block_cols
  val sp_mask_len = (spad_w / (aligned_to * 8)) max 1

  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()

  val reader = LazyModule(new StreamReader(config, max_in_flight_mem_reqs, dataBits, maxBytes, spad_w, acc_w, aligned_to,
    sp_banks * sp_bank_entries, acc_banks * acc_bank_entries, block_rows, use_tlb_register_filter,
    use_firesim_simulation_counters))
  val writer = LazyModule(new StreamWriter(max_in_flight_mem_reqs, dataBits, maxBytes,
    if (acc_read_full_width) acc_w else spad_w, aligned_to, inputType, block_cols, use_tlb_register_filter,
    use_firesim_simulation_counters))

  // TODO make a cross-bar vs two separate ports a config option
  // id_node :=* reader.node
  // id_node :=* writer.node

  xbar_node := TLBuffer() := reader.node // TODO
  xbar_node := TLBuffer() := writer.node
  id_node := TLWidthWidget(config.dma_buswidth/8) := TLBuffer() := xbar_node

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with HasCoreParameters {
    val acc_row_t = Vec(meshColumns, Vec(tileColumns, accType))
    val spad_row_t = Vec(meshColumns, Vec(tileColumns, inputType))

    val io = IO(new Bundle {
      // DMA ports
      val dma = new Bundle {
        val read = Flipped(new ScratchpadReadMemIO(local_addr_t, mvin_scale_t_bits))
        val write = Flipped(new ScratchpadWriteMemIO(local_addr_t, accType.getWidth, acc_scale_t_bits))
      }

      // SRAM ports
      val srams = new Bundle {
        val read = Flipped(Vec(sp_banks, new ScratchpadReadIO(sp_bank_entries, spad_w)))
        // changed
        val write = Flipped(Vec(sp_banks, new ScratchpadWriteIO(sp_bank_entries, spad_w, (spad_w / (aligned_to * 8)) max 1)))
        // val write = Flipped(Vec(sp_banks, Decoupled(new ScratchpadWriteIO(sp_bank_entries, spad_w, (spad_w / (aligned_to * 8)) max 1))))
      }

      // Accumulator ports
      val acc = new Bundle {
        val read_req = Flipped(Vec(acc_banks, Decoupled(new AccumulatorReadReq(
          acc_bank_entries, accType, acc_scale_t.asInstanceOf[V]
        ))))
        val read_resp = Vec(acc_banks, Decoupled(new AccumulatorScaleResp(
          Vec(meshColumns, Vec(tileColumns, inputType)),
          Vec(meshColumns, Vec(tileColumns, accType))
        )))
        val write = Flipped(Vec(acc_banks, Decoupled(new AccumulatorWriteReq(
          acc_bank_entries, Vec(meshColumns, Vec(tileColumns, accType))
        ))))
      }

      val exwrite_grant = if (use_shared_ext_mem) Some(new Bundle {
        val spad = Flipped(Vec(sp_banks, Decoupled(new BankExWriteGrantReq(sp_bank_entries))))
        val acc = Flipped(Vec(acc_banks, Decoupled(new BankExWriteGrantReq(acc_bank_entries))))
      }) else None

      val exwrite_remind = if (use_shared_ext_mem) Some(new Bundle {
        val spad = Flipped(Vec(sp_banks, Decoupled(new BankExWriteRemindReq(sp_bank_entries))))
        val acc = Flipped(Vec(acc_banks, Decoupled(new BankExWriteRemindReq(acc_bank_entries))))
      }) else None

      val ext_mem = if (use_shared_ext_mem) {
        // changed
        // Some(new ExtSpadMemIO(sp_banks, acc_banks, acc_sub_banks))
        Some(new ExtMemIO_new(sp_banks, sp_sub_banks, sp_bank_entries / sp_sub_banks, spad_w, sp_mask_len, 8, acc_banks, acc_sub_banks, acc_bank_entries / acc_sub_banks, acc_row_t, 8))
      } else {
        None
      }

      // TLB ports
      val tlb = Vec(2, new FrontendTLBIO)

      // Misc. ports
      val busy = Output(Bool())
      val flush = Input(Bool())
      val counter = new CounterEventIO()
    })

    val write_dispatch_q = Queue(io.dma.write.req)
    // Write norm/scale queues are necessary to maintain in-order requests to accumulator norm/scale units
    // Writes from main SPAD just flow directly between scale_q and issue_q, while writes
    // From acc are ordered
    val write_norm_q = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits), spad_read_delay+2))
    val write_scale_q = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits), spad_read_delay+2))
    val write_issue_q = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits), spad_read_delay+1, pipe=true))
    val read_issue_q = Module(new Queue(new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits), spad_read_delay+1, pipe=true)) // TODO can't this just be a normal queue?

    write_dispatch_q.ready := false.B

    write_norm_q.io.enq.valid := false.B
    write_norm_q.io.enq.bits := write_dispatch_q.bits
    write_norm_q.io.deq.ready := false.B

    write_scale_q.io.enq.valid := false.B
    write_scale_q.io.enq.bits  := write_norm_q.io.deq.bits
    write_scale_q.io.deq.ready := false.B

    write_issue_q.io.enq.valid := false.B
    write_issue_q.io.enq.bits := write_scale_q.io.deq.bits

    // Garbage can immediately fire from dispatch_q -> norm_q
    when (write_dispatch_q.bits.laddr.is_garbage()) {
      write_norm_q.io.enq <> write_dispatch_q
    }

    // Non-acc or garbage can immediately fire between norm_q and scale_q
    when (write_norm_q.io.deq.bits.laddr.is_garbage() || !write_norm_q.io.deq.bits.laddr.is_acc_addr) {
      write_scale_q.io.enq <> write_norm_q.io.deq
    }

    // Non-acc or garbage can immediately fire between scale_q and issue_q
    when (write_scale_q.io.deq.bits.laddr.is_garbage() || !write_scale_q.io.deq.bits.laddr.is_acc_addr) {
      write_issue_q.io.enq <> write_scale_q.io.deq
    }

    val writeData = Wire(Valid(UInt((spad_w max acc_w).W)))
    writeData.valid := write_issue_q.io.deq.bits.laddr.is_garbage()
    writeData.bits := DontCare
    val fullAccWriteData = Wire(UInt(acc_w.W))
    fullAccWriteData := DontCare
    val writeData_is_full_width = !write_issue_q.io.deq.bits.laddr.is_garbage() &&
      write_issue_q.io.deq.bits.laddr.is_acc_addr && write_issue_q.io.deq.bits.laddr.read_full_acc_row
    val writeData_is_all_zeros = write_issue_q.io.deq.bits.laddr.is_garbage()

    writer.module.io.req.valid := write_issue_q.io.deq.valid && writeData.valid
    write_issue_q.io.deq.ready := writer.module.io.req.ready && writeData.valid
    writer.module.io.req.bits.vaddr := write_issue_q.io.deq.bits.vaddr
    writer.module.io.req.bits.len := Mux(writeData_is_full_width,
      write_issue_q.io.deq.bits.len * (accType.getWidth / 8).U,
      write_issue_q.io.deq.bits.len * (inputType.getWidth / 8).U)
    writer.module.io.req.bits.data := MuxCase(writeData.bits, Seq(
       writeData_is_all_zeros -> 0.U,
       writeData_is_full_width -> fullAccWriteData
    ))
    writer.module.io.req.bits.block := write_issue_q.io.deq.bits.block
    writer.module.io.req.bits.status := write_issue_q.io.deq.bits.status
    writer.module.io.req.bits.pool_en := write_issue_q.io.deq.bits.pool_en
    writer.module.io.req.bits.store_en := write_issue_q.io.deq.bits.store_en

    io.dma.write.resp.valid := false.B
    io.dma.write.resp.bits.cmd_id := write_dispatch_q.bits.cmd_id
    when (write_dispatch_q.bits.laddr.is_garbage() && write_dispatch_q.fire) {
      io.dma.write.resp.valid := true.B
    }

    read_issue_q.io.enq <> io.dma.read.req

    val zero_writer = Module(new ZeroWriter(config, new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits)))

    when (io.dma.read.req.bits.all_zeros) {
      read_issue_q.io.enq.valid := false.B
      io.dma.read.req.ready := zero_writer.io.req.ready
    }

    zero_writer.io.req.valid := io.dma.read.req.valid && io.dma.read.req.bits.all_zeros
    zero_writer.io.req.bits.laddr := io.dma.read.req.bits.laddr
    zero_writer.io.req.bits.cols := io.dma.read.req.bits.cols
    zero_writer.io.req.bits.block_stride := io.dma.read.req.bits.block_stride
    zero_writer.io.req.bits.tag := io.dma.read.req.bits

    val zero_writer_pixel_repeater = Module(new PixelRepeater(inputType, local_addr_t, block_cols, aligned_to, new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits), passthrough = !has_first_layer_optimizations))
    zero_writer_pixel_repeater.io.req.valid := zero_writer.io.resp.valid
    zero_writer_pixel_repeater.io.req.bits.in := 0.U.asTypeOf(Vec(block_cols, inputType))
    zero_writer_pixel_repeater.io.req.bits.laddr := zero_writer.io.resp.bits.laddr
    zero_writer_pixel_repeater.io.req.bits.len := zero_writer.io.resp.bits.tag.cols
    zero_writer_pixel_repeater.io.req.bits.pixel_repeats := zero_writer.io.resp.bits.tag.pixel_repeats
    zero_writer_pixel_repeater.io.req.bits.last := zero_writer.io.resp.bits.last
    zero_writer_pixel_repeater.io.req.bits.tag := zero_writer.io.resp.bits.tag
    zero_writer_pixel_repeater.io.req.bits.mask := {
      val n = inputType.getWidth / 8
      val mask = zero_writer.io.resp.bits.mask
      val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
      expanded
    }

    zero_writer.io.resp.ready := zero_writer_pixel_repeater.io.req.ready
    zero_writer_pixel_repeater.io.resp.ready := false.B

    reader.module.io.req.valid := read_issue_q.io.deq.valid
    read_issue_q.io.deq.ready := reader.module.io.req.ready
    reader.module.io.req.bits.vaddr := read_issue_q.io.deq.bits.vaddr
    reader.module.io.req.bits.spaddr := Mux(read_issue_q.io.deq.bits.laddr.is_acc_addr,
      read_issue_q.io.deq.bits.laddr.full_acc_addr(), read_issue_q.io.deq.bits.laddr.full_sp_addr())
    reader.module.io.req.bits.len := read_issue_q.io.deq.bits.cols
    reader.module.io.req.bits.repeats := read_issue_q.io.deq.bits.repeats
    reader.module.io.req.bits.pixel_repeats := read_issue_q.io.deq.bits.pixel_repeats
    reader.module.io.req.bits.scale := read_issue_q.io.deq.bits.scale
    reader.module.io.req.bits.is_acc := read_issue_q.io.deq.bits.laddr.is_acc_addr
    reader.module.io.req.bits.accumulate := read_issue_q.io.deq.bits.laddr.accumulate
    reader.module.io.req.bits.has_acc_bitwidth := read_issue_q.io.deq.bits.has_acc_bitwidth
    reader.module.io.req.bits.block_stride := read_issue_q.io.deq.bits.block_stride
    reader.module.io.req.bits.status := read_issue_q.io.deq.bits.status
    reader.module.io.req.bits.cmd_id := read_issue_q.io.deq.bits.cmd_id

    val (mvin_scale_in, mvin_scale_out) = VectorScalarMultiplier(
      config.mvin_scale_args,
      config.inputType, config.meshColumns * config.tileColumns, chiselTypeOf(reader.module.io.resp.bits),
      is_acc = false
    )
    val (mvin_scale_acc_in, mvin_scale_acc_out) = if (mvin_scale_shared) (mvin_scale_in, mvin_scale_out) else (
      VectorScalarMultiplier(
        config.mvin_scale_acc_args,
        config.accType, config.meshColumns * config.tileColumns, chiselTypeOf(reader.module.io.resp.bits),
        is_acc = true
      )
    )

    mvin_scale_in.valid := reader.module.io.resp.valid && (mvin_scale_shared.B || !reader.module.io.resp.bits.is_acc ||
      (reader.module.io.resp.bits.is_acc && !reader.module.io.resp.bits.has_acc_bitwidth))

    mvin_scale_in.bits.in := reader.module.io.resp.bits.data.asTypeOf(chiselTypeOf(mvin_scale_in.bits.in))
    mvin_scale_in.bits.scale := reader.module.io.resp.bits.scale.asTypeOf(mvin_scale_t)
    mvin_scale_in.bits.repeats := reader.module.io.resp.bits.repeats
    mvin_scale_in.bits.pixel_repeats := reader.module.io.resp.bits.pixel_repeats
    mvin_scale_in.bits.last := reader.module.io.resp.bits.last
    mvin_scale_in.bits.tag := reader.module.io.resp.bits

    val mvin_scale_pixel_repeater = Module(new PixelRepeater(inputType, local_addr_t, block_cols, aligned_to, mvin_scale_out.bits.tag.cloneType, passthrough = !has_first_layer_optimizations))
    mvin_scale_pixel_repeater.io.req.valid := mvin_scale_out.valid
    mvin_scale_pixel_repeater.io.req.bits.in := mvin_scale_out.bits.out
    mvin_scale_pixel_repeater.io.req.bits.mask := mvin_scale_out.bits.tag.mask take mvin_scale_pixel_repeater.io.req.bits.mask.size
    mvin_scale_pixel_repeater.io.req.bits.laddr := mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
    mvin_scale_pixel_repeater.io.req.bits.len := mvin_scale_out.bits.tag.len
    mvin_scale_pixel_repeater.io.req.bits.pixel_repeats := mvin_scale_out.bits.tag.pixel_repeats
    mvin_scale_pixel_repeater.io.req.bits.last := mvin_scale_out.bits.last
    mvin_scale_pixel_repeater.io.req.bits.tag := mvin_scale_out.bits.tag

    mvin_scale_out.ready := mvin_scale_pixel_repeater.io.req.ready
    mvin_scale_pixel_repeater.io.resp.ready := false.B

    if (!mvin_scale_shared) {
      mvin_scale_acc_in.valid := reader.module.io.resp.valid &&
        (reader.module.io.resp.bits.is_acc && reader.module.io.resp.bits.has_acc_bitwidth)
      mvin_scale_acc_in.bits.in := reader.module.io.resp.bits.data.asTypeOf(chiselTypeOf(mvin_scale_acc_in.bits.in))
      mvin_scale_acc_in.bits.scale := reader.module.io.resp.bits.scale.asTypeOf(mvin_scale_acc_t)
      mvin_scale_acc_in.bits.repeats := reader.module.io.resp.bits.repeats
      mvin_scale_acc_in.bits.pixel_repeats := 1.U
      mvin_scale_acc_in.bits.last := reader.module.io.resp.bits.last
      mvin_scale_acc_in.bits.tag := reader.module.io.resp.bits

      mvin_scale_acc_out.ready := false.B
    }

    reader.module.io.resp.ready := Mux(reader.module.io.resp.bits.is_acc && reader.module.io.resp.bits.has_acc_bitwidth,
      mvin_scale_acc_in.ready, mvin_scale_in.ready)

    val mvin_scale_finished = mvin_scale_pixel_repeater.io.resp.fire && mvin_scale_pixel_repeater.io.resp.bits.last
    val mvin_scale_acc_finished = mvin_scale_acc_out.fire && mvin_scale_acc_out.bits.last
    val zero_writer_finished = zero_writer_pixel_repeater.io.resp.fire && zero_writer_pixel_repeater.io.resp.bits.last

    val zero_writer_bytes_read = Mux(zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr,
      zero_writer_pixel_repeater.io.resp.bits.tag.cols * (accType.getWidth / 8).U,
      zero_writer_pixel_repeater.io.resp.bits.tag.cols * (inputType.getWidth / 8).U)

    // For DMA read responses, mvin_scale gets first priority, then mvin_scale_acc, and then zero_writer
    io.dma.read.resp.valid := mvin_scale_finished || mvin_scale_acc_finished || zero_writer_finished

    // io.dma.read.resp.bits.cmd_id := MuxCase(zero_writer.io.resp.bits.tag.cmd_id, Seq(
    io.dma.read.resp.bits.cmd_id := MuxCase(zero_writer_pixel_repeater.io.resp.bits.tag.cmd_id, Seq(
      // mvin_scale_finished -> mvin_scale_out.bits.tag.cmd_id,
      mvin_scale_finished -> mvin_scale_pixel_repeater.io.resp.bits.tag.cmd_id,
      mvin_scale_acc_finished -> mvin_scale_acc_out.bits.tag.cmd_id))

    io.dma.read.resp.bits.bytesRead := MuxCase(zero_writer_bytes_read, Seq(
      // mvin_scale_finished -> mvin_scale_out.bits.tag.bytes_read,
      mvin_scale_finished -> mvin_scale_pixel_repeater.io.resp.bits.tag.bytes_read,
      mvin_scale_acc_finished -> mvin_scale_acc_out.bits.tag.bytes_read))

    io.tlb(0) <> writer.module.io.tlb
    io.tlb(1) <> reader.module.io.tlb

    writer.module.io.flush := io.flush
    reader.module.io.flush := io.flush

    io.busy := writer.module.io.busy || reader.module.io.busy || write_issue_q.io.deq.valid || write_norm_q.io.deq.valid || write_scale_q.io.deq.valid || write_dispatch_q.valid

    val spad_mems = if (!use_shared_ext_mem) {
      val banks = Seq.fill(sp_banks) { Module(new ScratchpadBank(
        sp_bank_entries, spad_w,
        aligned_to, config.sp_singleported,
        use_shared_ext_mem, is_dummy
      )) }
      val bank_ios = VecInit(banks.map(_.io))
      // Reading from the SRAM banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        if (use_shared_ext_mem) {
          io.ext_mem.get.spad(i) <> bio.ext_mem.get
        }

        val ex_read_req = io.srams.read(i).req
        val exread = ex_read_req.valid

        // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
        val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
          !write_dispatch_q.bits.laddr.is_garbage() &&
          !(bio.write.en && config.sp_singleported.B) &&
          !write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.sp_bank() === i.U

        bio.read.req.valid := exread || dmawrite
        ex_read_req.ready := bio.read.req.ready

        // The ExecuteController gets priority when reading from SRAMs
        when (exread) {
          bio.read.req.bits.addr := ex_read_req.bits.addr
          bio.read.req.bits.fromDMA := false.B
        }.elsewhen (dmawrite) {
          bio.read.req.bits.addr := write_dispatch_q.bits.laddr.sp_row()
          bio.read.req.bits.fromDMA := true.B

          when (bio.read.req.fire) {
            write_dispatch_q.ready := true.B
            write_norm_q.io.enq.valid := true.B

            io.dma.write.resp.valid := true.B
          }
        }.otherwise {
          bio.read.req.bits := DontCare
        }

        val dma_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
        dma_read_resp.valid := bio.read.resp.valid && bio.read.resp.bits.fromDMA
        dma_read_resp.bits := bio.read.resp.bits
        val ex_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
        ex_read_resp.valid := bio.read.resp.valid && !bio.read.resp.bits.fromDMA
        ex_read_resp.bits := bio.read.resp.bits

        val dma_read_pipe = Pipeline(dma_read_resp, spad_read_delay)
        val ex_read_pipe = Pipeline(ex_read_resp, spad_read_delay)

        bio.read.resp.ready := Mux(bio.read.resp.bits.fromDMA, dma_read_resp.ready, ex_read_resp.ready)

        dma_read_pipe.ready := writer.module.io.req.ready &&
          !write_issue_q.io.deq.bits.laddr.is_acc_addr && write_issue_q.io.deq.bits.laddr.sp_bank() === i.U && // I believe we don't need to check that write_issue_q is valid here, because if the SRAM's resp is valid, then that means that the write_issue_q's deq should also be valid
          !write_issue_q.io.deq.bits.laddr.is_garbage()
        when (dma_read_pipe.fire) {
          writeData.valid := true.B
          writeData.bits := dma_read_pipe.bits.data
        }

        io.srams.read(i).resp <> ex_read_pipe
      }

      // Writing to the SRAM banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        val exwrite = io.srams.write(i).en

        // val laddr = mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
        val laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr

        // val dmaread = mvin_scale_out.valid && !mvin_scale_out.bits.tag.is_acc &&
        val dmaread = mvin_scale_pixel_repeater.io.resp.valid && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc &&
          laddr.sp_bank() === i.U


        // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
        // or mvin_acc_scale at the same time. The scalers always get priority in those cases
        /* val zerowrite = zero_writer.io.resp.valid && !zero_writer.io.resp.bits.laddr.is_acc_addr &&
          zero_writer.io.resp.bits.laddr.sp_bank() === i.U && */
        val zerowrite = zero_writer_pixel_repeater.io.resp.valid && !zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
          zero_writer_pixel_repeater.io.resp.bits.laddr.sp_bank() === i.U &&
          // !((mvin_scale_out.valid && mvin_scale_out.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))
          !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

        bio.write.en := exwrite || dmaread || zerowrite

        when (exwrite) {
          bio.write.addr := io.srams.write(i).addr
          bio.write.data := io.srams.write(i).data
          bio.write.mask := io.srams.write(i).mask
        }.elsewhen (dmaread) {
          bio.write.addr := laddr.sp_row()
          bio.write.data := mvin_scale_pixel_repeater.io.resp.bits.out.asUInt
          bio.write.mask := mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)

          mvin_scale_pixel_repeater.io.resp.ready := true.B // TODO we combinationally couple valid and ready signals
        }.elsewhen (zerowrite) {
          bio.write.addr := zero_writer_pixel_repeater.io.resp.bits.laddr.sp_row()
          bio.write.data := 0.U
          bio.write.mask := zero_writer_pixel_repeater.io.resp.bits.mask

          zero_writer_pixel_repeater.io.resp.ready := true.B // TODO we combinationally couple valid and ready signals
        }.otherwise {
          bio.write.addr := DontCare
          bio.write.data := DontCare
          bio.write.mask := DontCare
        }
      }
      Some(banks)
    }
    else {
      // Reading from the SRAM banks
      val spad_adapters = Seq.fill(sp_banks) {
        Module(new ExtSpadSubBankAdapter(sp_bank_entries, sp_sub_banks, spad_w, sp_mask_len, 2))
      }
      spad_adapters.zipWithIndex.foreach { case (adapter, i) =>
        io.ext_mem.get.spad(i) <> adapter.io.ext
      }
      val bank_ios = spad_adapters.map(_.io.bank)
      bank_ios.zip(io.exwrite_grant.get.spad).foreach { case (bio, grant) =>
        bio.grant <> grant
      }
      bank_ios.zip(io.exwrite_remind.get.spad).foreach { case (bio, remind) =>
        bio.remind <> remind
      }
      bank_ios.zipWithIndex.foreach { case (bio, i) =>

        val ex_read_req = io.srams.read(i).req
        val exread = ex_read_req.valid

        // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
        val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
          !write_dispatch_q.bits.laddr.is_garbage() &&
          !(bio.write.valid && config.sp_singleported.B) &&
          !write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.sp_bank() === i.U

        bio.read.req.valid := exread || dmawrite
        ex_read_req.ready := bio.read.req.ready

        // The ExecuteController gets priority when reading from SRAMs
        when (exread) {
          bio.read.req.bits.addr := ex_read_req.bits.addr
          bio.read.req.bits.fromDMA := false.B
        }.elsewhen (dmawrite) {
          bio.read.req.bits.addr := write_dispatch_q.bits.laddr.sp_row()
          bio.read.req.bits.fromDMA := true.B

          when (bio.read.req.fire) {
            write_dispatch_q.ready := true.B
            write_norm_q.io.enq.valid := true.B

            io.dma.write.resp.valid := true.B
          }
        }.otherwise {
          bio.read.req.bits := DontCare
        }

        val dma_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
        dma_read_resp.valid := bio.read.resp.valid && bio.read.resp.bits.fromDMA
        dma_read_resp.bits := bio.read.resp.bits
        val ex_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
        ex_read_resp.valid := bio.read.resp.valid && !bio.read.resp.bits.fromDMA
        ex_read_resp.bits := bio.read.resp.bits

        val dma_read_pipe = Pipeline(dma_read_resp, spad_read_delay)
        val ex_read_pipe = Pipeline(ex_read_resp, spad_read_delay)

        bio.read.resp.ready := Mux(bio.read.resp.bits.fromDMA, dma_read_resp.ready, ex_read_resp.ready)

        dma_read_pipe.ready := writer.module.io.req.ready &&
          !write_issue_q.io.deq.bits.laddr.is_acc_addr && write_issue_q.io.deq.bits.laddr.sp_bank() === i.U && // I believe we don't need to check that write_issue_q is valid here, because if the SRAM's resp is valid, then that means that the write_issue_q's deq should also be valid
          !write_issue_q.io.deq.bits.laddr.is_garbage()
        when (dma_read_pipe.fire) {
          writeData.valid := true.B
          writeData.bits := dma_read_pipe.bits.data
        }

        io.srams.read(i).resp <> ex_read_pipe
      }

      // Writing to the SRAM banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        val exwrite = io.srams.write(i).en

        // val laddr = mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
        val laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr

        // val dmaread = mvin_scale_out.valid && !mvin_scale_out.bits.tag.is_acc &&
        val dmaread = mvin_scale_pixel_repeater.io.resp.valid && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc &&
          laddr.sp_bank() === i.U


        // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
        // or mvin_acc_scale at the same time. The scalers always get priority in those cases
        /* val zerowrite = zero_writer.io.resp.valid && !zero_writer.io.resp.bits.laddr.is_acc_addr &&
          zero_writer.io.resp.bits.laddr.sp_bank() === i.U && */
        val zerowrite = zero_writer_pixel_repeater.io.resp.valid && !zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
          zero_writer_pixel_repeater.io.resp.bits.laddr.sp_bank() === i.U &&
          // !((mvin_scale_out.valid && mvin_scale_out.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))
          !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

        bio.write.valid := exwrite || dmaread || zerowrite

        assert(!(exwrite && !bio.write.ready))

        when (exwrite) {
          bio.write.bits.addr := io.srams.write(i).addr
          bio.write.bits.data := io.srams.write(i).data
          bio.write.bits.mask := io.srams.write(i).mask
          bio.write.bits.exwrite := true.B
        }.elsewhen (dmaread) {
          bio.write.bits.addr := laddr.sp_row()
          bio.write.bits.data := mvin_scale_pixel_repeater.io.resp.bits.out.asUInt
          bio.write.bits.mask := mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)
          bio.write.bits.exwrite := false.B

          mvin_scale_pixel_repeater.io.resp.ready := bio.write.fire // TODO we combinationally couple valid and ready signals
        }.elsewhen (zerowrite) {
          bio.write.bits.addr := zero_writer_pixel_repeater.io.resp.bits.laddr.sp_row()
          bio.write.bits.data := 0.U
          bio.write.bits.mask := zero_writer_pixel_repeater.io.resp.bits.mask
          bio.write.bits.exwrite := false.B

          zero_writer_pixel_repeater.io.resp.ready := bio.write.fire // TODO we combinationally couple valid and ready signals
        }.otherwise {
          bio.write.bits.addr := DontCare
          bio.write.bits.data := DontCare
          bio.write.bits.mask := DontCare
          bio.write.bits.exwrite := DontCare
        }
      }
      None
    }

//    val acc_norm_unit = Module(new Normalizer(
//      max_len = block_cols,
//      num_reduce_lanes = -1,
//      num_stats = 4,
//      latency = 4,
//      fullDataType = acc_row_t,
//      scale_t = acc_scale_t,
//    ))

    val (acc_norm_unit_in, acc_norm_unit_out) = Normalizer(
      is_passthru = !config.has_normalizations,
      max_len = block_cols,
      num_reduce_lanes = -1,
      num_stats = 4,
      latency = 4,
      fullDataType = acc_row_t,
      scale_t = acc_scale_t,
    )

    acc_norm_unit_in.valid := false.B
    acc_norm_unit_in.bits.len := write_norm_q.io.deq.bits.len
    acc_norm_unit_in.bits.stats_id := write_norm_q.io.deq.bits.acc_norm_stats_id
    acc_norm_unit_in.bits.cmd := write_norm_q.io.deq.bits.laddr.norm_cmd
    acc_norm_unit_in.bits.acc_read_resp := DontCare

    val acc_scale_unit = Module(new AccumulatorScale(
      acc_row_t,
      spad_row_t,
      acc_scale_t.asInstanceOf[V],
      acc_read_small_width,
      acc_read_full_width,
      acc_scale_func,
      acc_scale_num_units,
      acc_scale_latency,
      has_nonlinear_activations,
      has_normalizations,
    ))

    val acc_waiting_to_be_scaled = write_scale_q.io.deq.valid &&
      !write_scale_q.io.deq.bits.laddr.is_garbage() &&
      write_scale_q.io.deq.bits.laddr.is_acc_addr &&
      write_issue_q.io.enq.ready

    acc_norm_unit_out.ready := acc_scale_unit.io.in.ready && acc_waiting_to_be_scaled
    acc_scale_unit.io.in.valid := acc_norm_unit_out.valid && acc_waiting_to_be_scaled
    acc_scale_unit.io.in.bits  := acc_norm_unit_out.bits

    when (acc_scale_unit.io.in.fire()) {
      write_issue_q.io.enq <> write_scale_q.io.deq
    }

    acc_scale_unit.io.out.ready := false.B

    val dma_resp_ready =
      writer.module.io.req.ready &&
        write_issue_q.io.deq.bits.laddr.is_acc_addr &&
        !write_issue_q.io.deq.bits.laddr.is_garbage()

    when (acc_scale_unit.io.out.bits.fromDMA && dma_resp_ready) {
      // Send the acc-scale result into the DMA
      acc_scale_unit.io.out.ready := true.B
      writeData.valid := acc_scale_unit.io.out.valid
      writeData.bits  := acc_scale_unit.io.out.bits.data.asUInt
      fullAccWriteData := acc_scale_unit.io.out.bits.full_data.asUInt
    }
    for (i <- 0 until acc_banks) {
      // Send the acc-sccale result to the ExController
      io.acc.read_resp(i).valid := false.B
      io.acc.read_resp(i).bits  := acc_scale_unit.io.out.bits
      when (!acc_scale_unit.io.out.bits.fromDMA && acc_scale_unit.io.out.bits.acc_bank_id === i.U) {
        acc_scale_unit.io.out.ready := io.acc.read_resp(i).ready
        io.acc.read_resp(i).valid := acc_scale_unit.io.out.valid
      }
    }

    val acc_adders = if (use_shared_ext_mem) {
      None
    } else {
      Some(Module(new AccPipeShared(acc_latency-1, acc_row_t, acc_banks)))
    }

    val acc_mems = if (!use_shared_ext_mem) {
      val banks = Seq.fill(acc_banks) { Module(new AccumulatorMem(
        acc_bank_entries, acc_row_t, acc_scale_func, acc_scale_t.asInstanceOf[V],
        acc_singleported, acc_sub_banks,
        use_shared_ext_mem,
        acc_latency, accType, is_dummy
      )) }
      val bank_ios = VecInit(banks.map(_.io))

      // Getting the output of the bank that's about to be issued to the writer
      val bank_issued_io = bank_ios(write_issue_q.io.deq.bits.laddr.acc_bank())

      // Reading from the Accumulator banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        if (use_shared_ext_mem) {
          io.ext_mem.get.acc(i) <> bio.ext_mem.get
        }

        acc_adders.get.io.in_sel(i) := bio.adder.valid
        acc_adders.get.io.ina(i) := bio.adder.op1
        acc_adders.get.io.inb(i) := bio.adder.op2
        bio.adder.sum := acc_adders.get.io.out

        val ex_read_req = io.acc.read_req(i)
        val exread = ex_read_req.valid

        // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
        val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
          !write_dispatch_q.bits.laddr.is_garbage() &&
          write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.acc_bank() === i.U

        bio.read.req.valid := exread || dmawrite
        ex_read_req.ready := bio.read.req.ready

        // The ExecuteController gets priority when reading from accumulator banks
        when (exread) {
          bio.read.req.bits.addr := ex_read_req.bits.addr
          bio.read.req.bits.act := ex_read_req.bits.act
          bio.read.req.bits.igelu_qb := ex_read_req.bits.igelu_qb
          bio.read.req.bits.igelu_qc := ex_read_req.bits.igelu_qc
          bio.read.req.bits.iexp_qln2 := ex_read_req.bits.iexp_qln2
          bio.read.req.bits.iexp_qln2_inv := ex_read_req.bits.iexp_qln2_inv
          bio.read.req.bits.scale := ex_read_req.bits.scale
          bio.read.req.bits.full := false.B
          bio.read.req.bits.fromDMA := false.B
        }.elsewhen (dmawrite) {
          bio.read.req.bits.addr := write_dispatch_q.bits.laddr.acc_row()
          bio.read.req.bits.full := write_dispatch_q.bits.laddr.read_full_acc_row
          bio.read.req.bits.act := write_dispatch_q.bits.acc_act
          bio.read.req.bits.igelu_qb := write_dispatch_q.bits.acc_igelu_qb.asTypeOf(bio.read.req.bits.igelu_qb)
          bio.read.req.bits.igelu_qc := write_dispatch_q.bits.acc_igelu_qc.asTypeOf(bio.read.req.bits.igelu_qc)
          bio.read.req.bits.iexp_qln2 := write_dispatch_q.bits.acc_iexp_qln2.asTypeOf(bio.read.req.bits.iexp_qln2)
          bio.read.req.bits.iexp_qln2_inv := write_dispatch_q.bits.acc_iexp_qln2_inv.asTypeOf(bio.read.req.bits.iexp_qln2_inv)
          bio.read.req.bits.scale := write_dispatch_q.bits.acc_scale.asTypeOf(bio.read.req.bits.scale)
          bio.read.req.bits.fromDMA := true.B

          when (bio.read.req.fire) {
            write_dispatch_q.ready := true.B
            write_norm_q.io.enq.valid := true.B

            io.dma.write.resp.valid := true.B
          }
        }.otherwise {
          bio.read.req.bits := DontCare
        }
        bio.read.resp.ready := false.B

        when (write_norm_q.io.deq.valid &&
          acc_norm_unit_in.ready &&
          bio.read.resp.valid &&
          write_scale_q.io.enq.ready &&
          write_norm_q.io.deq.bits.laddr.is_acc_addr &&
          !write_norm_q.io.deq.bits.laddr.is_garbage() &&
          write_norm_q.io.deq.bits.laddr.acc_bank() === i.U)
        {
          write_norm_q.io.deq.ready := true.B
          acc_norm_unit_in.valid := true.B
          bio.read.resp.ready := true.B

          // Some normalizer commands don't write to main memory, so they don't need to be passed on to the scaling units
          write_scale_q.io.enq.valid := NormCmd.writes_to_main_memory(write_norm_q.io.deq.bits.laddr.norm_cmd)

          acc_norm_unit_in.bits.acc_read_resp := bio.read.resp.bits
          acc_norm_unit_in.bits.acc_read_resp.acc_bank_id := i.U
        }
      }

      // Writing to the accumulator banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        // Order of precedence during writes is ExecuteController, and then mvin_scale, and then mvin_scale_acc, and
        // then zero_writer

        val exwrite = io.acc.write(i).valid
        io.acc.write(i).ready := true.B
        assert(!(exwrite && !bio.write.ready), "Execute controller write to AccumulatorMem was skipped")

        // val from_mvin_scale = mvin_scale_out.valid && mvin_scale_out.bits.tag.is_acc
        val from_mvin_scale = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc
        val from_mvin_scale_acc = mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.tag.is_acc

        // val mvin_scale_laddr = mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
        val mvin_scale_laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr
        val mvin_scale_acc_laddr = mvin_scale_acc_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_acc_out.bits.row

        val dmaread_bank = Mux(from_mvin_scale, mvin_scale_laddr.acc_bank(),
          mvin_scale_acc_laddr.acc_bank())
        val dmaread_row = Mux(from_mvin_scale, mvin_scale_laddr.acc_row(), mvin_scale_acc_laddr.acc_row())

        // We need to make sure that we don't try to return a dma read resp from both mvin_scale and mvin_scale_acc
        // at the same time. mvin_scale always gets priority in this cases
        val spad_last = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc

        val dmaread = (from_mvin_scale || from_mvin_scale_acc) &&
          dmaread_bank === i.U /* &&
          (mvin_scale_same.B || from_mvin_scale || !spad_dmaread_last) */

        // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
        // or mvin_acc_scale at the same time. The scalers always get priority in those cases
        /* val zerowrite = zero_writer.io.resp.valid && zero_writer.io.resp.bits.laddr.is_acc_addr &&
          zero_writer.io.resp.bits.laddr.acc_bank() === i.U && */
        val zerowrite = zero_writer_pixel_repeater.io.resp.valid && zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
          zero_writer_pixel_repeater.io.resp.bits.laddr.acc_bank() === i.U &&
          // !((mvin_scale_out.valid && mvin_scale_out.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))
          !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

        val consecutive_write_block = RegInit(false.B)
        if (acc_singleported) {
          val consecutive_write_sub_bank = RegInit(0.U((1 max log2Ceil(acc_sub_banks)).W))
          when (bio.write.fire && bio.write.bits.acc &&
            (bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0) === consecutive_write_sub_bank)) {
            consecutive_write_block := true.B
          } .elsewhen (bio.write.fire && bio.write.bits.acc) {
            consecutive_write_block := false.B
            consecutive_write_sub_bank := bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0)
          } .otherwise {
            consecutive_write_block := false.B
          }
        }
        bio.write.valid := false.B

        // bio.write.bits.acc := MuxCase(zero_writer.io.resp.bits.laddr.accumulate,
        bio.write.bits.acc := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.accumulate,
          Seq(exwrite -> io.acc.write(i).bits.acc,
            // from_mvin_scale -> mvin_scale_out.bits.tag.accumulate,
            from_mvin_scale -> mvin_scale_pixel_repeater.io.resp.bits.tag.accumulate,
            from_mvin_scale_acc -> mvin_scale_acc_out.bits.tag.accumulate))

        // bio.write.bits.addr := MuxCase(zero_writer.io.resp.bits.laddr.acc_row(),
        bio.write.bits.addr := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.acc_row(),
          Seq(exwrite -> io.acc.write(i).bits.addr,
            (from_mvin_scale || from_mvin_scale_acc) -> dmaread_row))

        when (exwrite) {
          bio.write.valid := true.B
          bio.write.bits.data := io.acc.write(i).bits.data
          bio.write.bits.mask := io.acc.write(i).bits.mask
        }.elsewhen (dmaread && !spad_last && !consecutive_write_block) {
          bio.write.valid := true.B
          bio.write.bits.data := Mux(from_mvin_scale,
            // VecInit(mvin_scale_out.bits.out.map(e => e.withWidthOf(accType))).asTypeOf(acc_row_t),
            VecInit(mvin_scale_pixel_repeater.io.resp.bits.out.map(e => e.withWidthOf(accType))).asTypeOf(acc_row_t),
            mvin_scale_acc_out.bits.out.asTypeOf(acc_row_t))
          bio.write.bits.mask :=
            Mux(from_mvin_scale,
              {
                val n = accType.getWidth / inputType.getWidth
                // val mask = mvin_scale_out.bits.tag.mask take ((spad_w / (aligned_to * 8)) max 1)
                val mask = mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)
                val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
                expanded
              },
              mvin_scale_acc_out.bits.tag.mask)

          when(from_mvin_scale) {
            mvin_scale_pixel_repeater.io.resp.ready := bio.write.ready
          }.otherwise {
            mvin_scale_acc_out.ready := bio.write.ready
          }
        }.elsewhen (zerowrite && !spad_last && !consecutive_write_block) {
          bio.write.valid := true.B
          bio.write.bits.data := 0.U.asTypeOf(acc_row_t)
          bio.write.bits.mask := {
            val n = accType.getWidth / inputType.getWidth
            val mask = zero_writer_pixel_repeater.io.resp.bits.mask
            val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
            expanded
          }

          zero_writer_pixel_repeater.io.resp.ready := bio.write.ready
        }.otherwise {
          bio.write.bits.data := DontCare
          bio.write.bits.mask := DontCare
        }
      }
      Some(banks)
    } else {
      val acc_adapters = Seq.fill(acc_banks) {
        Module(new ExtAccSubBankAdapter(acc_bank_entries, acc_row_t, acc_scale_t.asInstanceOf[V], acc_sub_banks, 2))
      }
      acc_adapters.zipWithIndex.foreach { case (adapter, i) =>
        io.ext_mem.get.acc(i) <> adapter.io.ext
      }
      val bank_ios = VecInit(acc_adapters.map(_.io.bank))
      (bank_ios zip io.exwrite_grant.get.acc).foreach { case (bio, grant) =>
        bio.grant <> grant
      }
      (bank_ios zip io.exwrite_remind.get.acc).foreach { case (bio, remind) =>
        bio.remind <> remind
      }

      // Getting the output of the bank that's about to be issued to the writer
      val bank_issued_io = bank_ios(write_issue_q.io.deq.bits.laddr.acc_bank())

      // Reading from the Accumulator banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        val ex_read_req = io.acc.read_req(i)
        val exread = ex_read_req.valid

        // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
        val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
          !write_dispatch_q.bits.laddr.is_garbage() &&
          write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.acc_bank() === i.U

        bio.read.req.valid := exread || dmawrite
        ex_read_req.ready := bio.read.req.ready

        // The ExecuteController gets priority when reading from accumulator banks
        when (exread) {
          bio.read.req.bits.addr := ex_read_req.bits.addr
          bio.read.req.bits.act := ex_read_req.bits.act
          bio.read.req.bits.igelu_qb := ex_read_req.bits.igelu_qb
          bio.read.req.bits.igelu_qc := ex_read_req.bits.igelu_qc
          bio.read.req.bits.iexp_qln2 := ex_read_req.bits.iexp_qln2
          bio.read.req.bits.iexp_qln2_inv := ex_read_req.bits.iexp_qln2_inv
          bio.read.req.bits.scale := ex_read_req.bits.scale
          bio.read.req.bits.full := false.B
          bio.read.req.bits.fromDMA := false.B
        }.elsewhen (dmawrite) {
          bio.read.req.bits.addr := write_dispatch_q.bits.laddr.acc_row()
          bio.read.req.bits.full := write_dispatch_q.bits.laddr.read_full_acc_row
          bio.read.req.bits.act := write_dispatch_q.bits.acc_act
          bio.read.req.bits.igelu_qb := write_dispatch_q.bits.acc_igelu_qb.asTypeOf(bio.read.req.bits.igelu_qb)
          bio.read.req.bits.igelu_qc := write_dispatch_q.bits.acc_igelu_qc.asTypeOf(bio.read.req.bits.igelu_qc)
          bio.read.req.bits.iexp_qln2 := write_dispatch_q.bits.acc_iexp_qln2.asTypeOf(bio.read.req.bits.iexp_qln2)
          bio.read.req.bits.iexp_qln2_inv := write_dispatch_q.bits.acc_iexp_qln2_inv.asTypeOf(bio.read.req.bits.iexp_qln2_inv)
          bio.read.req.bits.scale := write_dispatch_q.bits.acc_scale.asTypeOf(bio.read.req.bits.scale)
          bio.read.req.bits.fromDMA := true.B

          when (bio.read.req.fire) {
            write_dispatch_q.ready := true.B
            write_norm_q.io.enq.valid := true.B

            io.dma.write.resp.valid := true.B
          }
        }.otherwise {
          bio.read.req.bits := DontCare
        }
        bio.read.resp.ready := false.B

        when (write_norm_q.io.deq.valid &&
          acc_norm_unit_in.ready &&
          bio.read.resp.valid &&
          write_scale_q.io.enq.ready &&
          write_norm_q.io.deq.bits.laddr.is_acc_addr &&
          !write_norm_q.io.deq.bits.laddr.is_garbage() &&
          write_norm_q.io.deq.bits.laddr.acc_bank() === i.U)
        {
          write_norm_q.io.deq.ready := true.B
          acc_norm_unit_in.valid := true.B
          bio.read.resp.ready := true.B

          // Some normalizer commands don't write to main memory, so they don't need to be passed on to the scaling units
          write_scale_q.io.enq.valid := NormCmd.writes_to_main_memory(write_norm_q.io.deq.bits.laddr.norm_cmd)

          acc_norm_unit_in.bits.acc_read_resp := bio.read.resp.bits
          acc_norm_unit_in.bits.acc_read_resp.acc_bank_id := i.U
        }
      }

      // Writing to the accumulator banks
      bank_ios.zipWithIndex.foreach { case (bio, i) =>
        // Order of precedence during writes is ExecuteController, and then mvin_scale, and then mvin_scale_acc, and
        // then zero_writer

        val exwrite = io.acc.write(i).valid
        io.acc.write(i).ready := true.B
        assert(!(exwrite && !bio.write.ready), "Execute controller write to AccumulatorMem was skipped")

        // val from_mvin_scale = mvin_scale_out.valid && mvin_scale_out.bits.tag.is_acc
        val from_mvin_scale = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc
        val from_mvin_scale_acc = mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.tag.is_acc

        // val mvin_scale_laddr = mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
        val mvin_scale_laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr
        val mvin_scale_acc_laddr = mvin_scale_acc_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_acc_out.bits.row

        val dmaread_bank = Mux(from_mvin_scale, mvin_scale_laddr.acc_bank(),
          mvin_scale_acc_laddr.acc_bank())
        val dmaread_row = Mux(from_mvin_scale, mvin_scale_laddr.acc_row(), mvin_scale_acc_laddr.acc_row())

        // We need to make sure that we don't try to return a dma read resp from both mvin_scale and mvin_scale_acc
        // at the same time. mvin_scale always gets priority in this cases
        val spad_last = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc

        val dmaread = (from_mvin_scale || from_mvin_scale_acc) &&
          dmaread_bank === i.U /* &&
          (mvin_scale_same.B || from_mvin_scale || !spad_dmaread_last) */

        // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
        // or mvin_acc_scale at the same time. The scalers always get priority in those cases
        /* val zerowrite = zero_writer.io.resp.valid && zero_writer.io.resp.bits.laddr.is_acc_addr &&
          zero_writer.io.resp.bits.laddr.acc_bank() === i.U && */
        val zerowrite = zero_writer_pixel_repeater.io.resp.valid && zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
          zero_writer_pixel_repeater.io.resp.bits.laddr.acc_bank() === i.U &&
          // !((mvin_scale_out.valid && mvin_scale_out.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))
          !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

        val consecutive_write_block = RegInit(false.B)
        if (acc_singleported) {
          val consecutive_write_sub_bank = RegInit(0.U((1 max log2Ceil(acc_sub_banks)).W))
          when (bio.write.fire && bio.write.bits.acc &&
            (bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0) === consecutive_write_sub_bank)) {
            consecutive_write_block := true.B
          } .elsewhen (bio.write.fire && bio.write.bits.acc) {
            consecutive_write_block := false.B
            consecutive_write_sub_bank := bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0)
          } .otherwise {
            consecutive_write_block := false.B
          }
        }
        bio.write.valid := false.B

        assert(!(exwrite && !bio.write.ready))

        // bio.write.bits.acc := MuxCase(zero_writer.io.resp.bits.laddr.accumulate,
        bio.write.bits.acc := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.accumulate,
          Seq(exwrite -> io.acc.write(i).bits.acc,
            // from_mvin_scale -> mvin_scale_out.bits.tag.accumulate,
            from_mvin_scale -> mvin_scale_pixel_repeater.io.resp.bits.tag.accumulate,
            from_mvin_scale_acc -> mvin_scale_acc_out.bits.tag.accumulate))

        // bio.write.bits.addr := MuxCase(zero_writer.io.resp.bits.laddr.acc_row(),
        bio.write.bits.addr := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.acc_row(),
          Seq(exwrite -> io.acc.write(i).bits.addr,
            (from_mvin_scale || from_mvin_scale_acc) -> dmaread_row))

        when (exwrite) {
          bio.write.valid := true.B
          bio.write.bits.data := io.acc.write(i).bits.data
          bio.write.bits.mask := io.acc.write(i).bits.mask
          bio.write.bits.exwrite := true.B
        }.elsewhen (dmaread && !spad_last && !consecutive_write_block) {
          bio.write.valid := true.B
          bio.write.bits.data := Mux(from_mvin_scale,
            // VecInit(mvin_scale_out.bits.out.map(e => e.withWidthOf(accType))).asTypeOf(acc_row_t),
            VecInit(mvin_scale_pixel_repeater.io.resp.bits.out.map(e => e.withWidthOf(accType))).asTypeOf(acc_row_t),
            mvin_scale_acc_out.bits.out.asTypeOf(acc_row_t))
          bio.write.bits.mask :=
            Mux(from_mvin_scale,
              {
                val n = accType.getWidth / inputType.getWidth
                // val mask = mvin_scale_out.bits.tag.mask take ((spad_w / (aligned_to * 8)) max 1)
                val mask = mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)
                val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
                expanded
              },
              mvin_scale_acc_out.bits.tag.mask)
          bio.write.bits.exwrite := false.B

          when(from_mvin_scale) {
            mvin_scale_pixel_repeater.io.resp.ready := bio.write.fire
          }.otherwise {
            mvin_scale_acc_out.ready := bio.write.fire
          }
        }.elsewhen (zerowrite && !spad_last && !consecutive_write_block) {
          bio.write.valid := true.B
          bio.write.bits.data := 0.U.asTypeOf(acc_row_t)
          bio.write.bits.mask := {
            val n = accType.getWidth / inputType.getWidth
            val mask = zero_writer_pixel_repeater.io.resp.bits.mask
            val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
            expanded
          }
          bio.write.bits.exwrite := false.B

          zero_writer_pixel_repeater.io.resp.ready := bio.write.fire
        }.otherwise {
          bio.write.bits.data := DontCare
          bio.write.bits.mask := DontCare
          bio.write.bits.exwrite := DontCare
        }
      }
      None
    }

    // Counter connection
    io.counter.collect(reader.module.io.counter)
    io.counter.collect(writer.module.io.counter)
  }
}
