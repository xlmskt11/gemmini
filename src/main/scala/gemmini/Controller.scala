
package gemmini

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.util.ClockGate
import freechips.rocketchip.tilelink.TLIdentityNode
import GemminiISA._
import VpuLocalAddr._
import Util._

class GemminiCmd(rob_entries: Int)(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
  val rob_id = UDValid(UInt(log2Up(rob_entries).W))
  val from_matmul_fsm = Bool()
  val from_conv_fsm = Bool()
}

class Gemmini[T <: Data : Arithmetic, U <: Data, V <: Data](
    val config: GemminiArrayConfig[T, U, V],
    commandRoute: RoCCCommandRoute = RoCCCommandRoute.default)
                                     (implicit p: Parameters)
  extends LazyRoCC (
    opcodes = config.opcodes,
    nPTWPorts = if (config.use_shared_tlb) 1 else if (config.use_profiler) 3 else 2,
    commandRoute = commandRoute) {

  Files.write(Paths.get(config.headerFilePath), config.generateHeader().getBytes(StandardCharsets.UTF_8))
  if (System.getenv("GEMMINI_ONLY_GENERATE_GEMMINI_H") == "1") {
    System.exit(1)
  }

  val xLen = p(XLen)
  val spad = LazyModule(new Scratchpad(config))
  val profilers = if (config.use_profiler) {
    Some(LazyModule(new Profiler(config, new GemminiCmd(config.reservation_station_entries))))
  } else {
    None
  }

  override lazy val module = new GemminiModule(this)
  override val tlNode = if (config.use_dedicated_tl_port) {
    spad.id_node
  } else {
    profilers.map(_.id_node).getOrElse(TLIdentityNode())
  }
  override val atlNode = if (config.use_dedicated_tl_port) {
    profilers.map(_.id_node).getOrElse(TLIdentityNode())
  } else {
    spad.id_node
  }

  val node = if (config.use_dedicated_tl_port) tlNode else atlNode
}

class GemminiModule[T <: Data: Arithmetic, U <: Data, V <: Data]
    (outer: Gemmini[T, U, V])
    extends LazyRoCCModuleImp(outer)
    with HasCoreParameters {

  import outer.config._
  import outer.spad
  import outer.profilers

  // changed
  // val ext_mem_io = if (use_shared_ext_mem) Some(IO(new ExtSpadMemIO(sp_banks, acc_banks, acc_sub_banks))) else None
  val block_cols = meshColumns * tileColumns
  val spad_w = inputType.getWidth *  block_cols
  val sp_mask_len = (spad_w / (aligned_to * 8)) max 1
  val acc_row_t = Vec(meshColumns, Vec(tileColumns, accType))
  private val vpuRowAddrBits = if (use_vpu_fusion) {
    Some(1 max log2Ceil(acc_banks * acc_bank_entries))
  } else {
    None
  }
  val ext_mem_io = if (use_shared_ext_mem) Some(IO(new ExtMemIO_new(
    sp_banks, sp_sub_banks, sp_bank_entries / sp_sub_banks, spad_w, sp_mask_len,
    acc_banks, acc_sub_banks, acc_bank_entries / acc_sub_banks, acc_row_t,
    use_vpu_fusion
  ))) else None
  ext_mem_io.foreach(_ <> outer.spad.module.io.ext_mem.get)
  val vpu_matrix_read_io = if (use_vpu_fusion) Some(IO(
    new GemminiVpuMatrixReadIO(
      vpuRowAddrBits.get, block_cols, accType.getWidth))) else None
  val vpu_matrix_write_io = if (use_vpu_fusion && !use_shared_ext_mem) {
    Some(IO(Valid(new GemminiVpuMatrixWriteReq(
      vpuRowAddrBits.get, block_cols, accType.getWidth))))
  } else {
    None
  }
  if (use_vpu_fusion && !use_shared_ext_mem) {
    vpu_matrix_write_io.get <> outer.spad.module.io.acc.vpu_write.get
  }

  val tagWidth = 32

  // Counters
  val counters = Module(new CounterController(outer.config.num_counter, outer.xLen))
  io.resp <> counters.io.out  // Counter access command will be committed immediately
  counters.io.event_io.external_values(0) := 0.U
  counters.io.event_io.event_signal(0) := false.B
  counters.io.in.valid := false.B
  counters.io.in.bits := DontCare
  counters.io.event_io.collect(spad.module.io.counter)

  // Profiler
  if (use_profiler) {
    val profiler = profilers.get
    ProfileEventIO.init(profiler.module.io.profile_io.event_io)
    profiler.module.io.profiler_vaddr_valid := false.B
    profiler.module.io.profiler_vaddr := 0.U
    profiler.module.io.profiler_status := DontCare
  }

  // TLB
  implicit val edge = outer.spad.id_node.edges.out.head
  val nTlbClients = if (use_profiler) 3 else 2
  val tlb = Module(new FrontendTLB(nTlbClients, tlb_size, dma_maxbytes, use_tlb_register_filter, use_firesim_simulation_counters, use_shared_tlb))
  tlb.io.clients(0) <> outer.spad.module.io.tlb(0)
  tlb.io.clients(1) <> outer.spad.module.io.tlb(1)
  if (use_profiler) {
    tlb.io.clients(2) <> profilers.get.module.io.tlb
  }

  tlb.io.exp.foreach(_.flush_skip := false.B)
  tlb.io.exp.foreach(_.flush_retry := false.B)

  io.ptw <> tlb.io.ptw

  counters.io.event_io.collect(tlb.io.counter)

  spad.module.io.flush := tlb.io.exp.map(_.flush()).reduce(_ || _)

  val clock_en_reg = RegInit(true.B)
  val gated_clock = if (clock_gate) ClockGate(clock, clock_en_reg, "gemmini_clock_gate") else clock
  outer.spad.module.clock := gated_clock

  /*
  //=========================================================================
  // Frontends: Incoming commands and ROB
  //=========================================================================

  // forward cmd to correct frontend. if the rob is busy, do not forward new
  // commands to tiler, and vice versa
  val is_cisc_mode = RegInit(false.B)

  val raw_cmd = Queue(io.cmd)
  val funct = raw_cmd.bits.inst.funct

  val is_cisc_funct = (funct === CISC_CONFIG) ||
                      (funct === ADDR_AB) ||
                      (funct === ADDR_CD) ||
                      (funct === SIZE_MN) ||
                      (funct === SIZE_K) ||
                      (funct === RPT_BIAS) ||
                      (funct === RESET) ||
                      (funct === COMPUTE_CISC)

  val raw_cisc_cmd = WireInit(raw_cmd)
  val raw_risc_cmd = WireInit(raw_cmd)
  raw_cisc_cmd.valid := false.B
  raw_risc_cmd.valid := false.B
  raw_cmd.ready := false.B

  //-------------------------------------------------------------------------
  // cisc
  val cmd_fsm = CmdFSM(outer.config)
  cmd_fsm.io.cmd <> raw_cisc_cmd
  val tiler = TilerController(outer.config)
  tiler.io.cmd_in <> cmd_fsm.io.tiler

  //-------------------------------------------------------------------------
  // risc
  val unrolled_cmd = LoopUnroller(raw_risc_cmd, outer.config.meshRows * outer.config.tileRows)
  */

  val reservation_station = withClock (gated_clock) { Module(new ReservationStation(outer.config, new GemminiCmd(reservation_station_entries))) }
  val ext_deps_io = if (use_shared_res_entries) Some(IO(new EntriesForDeps(
    local_addr_t, reservation_station_entries_ld,
    reservation_station_entries_ex, reservation_station_entries_st,
    res_max_per_type))) else None
  ext_deps_io.foreach(_ <> reservation_station.io.ext_deps.get)
  val vsram_deps_io = if (use_vpu_fusion) Some(IO(
    new VsramEntriesForDeps(local_addr_t, reservation_station_entries_ld,
      reservation_station_entries_ex, reservation_station_entries_st,
      res_max_per_type))) else None
  vsram_deps_io.foreach(_ <> reservation_station.io.vsram_deps.get)
  counters.io.event_io.collect(reservation_station.io.counter)

  if (use_profiler) {
    val profiler = profilers.get
    profiler.module.io.profile_io.issue_cmd <> reservation_station.io.profile.get.issue_cmd
    profiler.module.io.profile_io.event_io.collect(reservation_station.io.profile.get.event_io)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ROB_ALLOC, reservation_station.io.profile.get.issue_cmd.fire(), reservation_station.io.profile.get.issue_cmd.rob_id)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ROB_ISSUE_LD, reservation_station.io.issue.ld.fire(), reservation_station.io.issue.ld.rob_id)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ROB_ISSUE_EX, reservation_station.io.issue.ex.fire(), reservation_station.io.issue.ex.rob_id)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ROB_ISSUE_ST, reservation_station.io.issue.st.fire(), reservation_station.io.issue.st.rob_id)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ROB_COMPLETE, reservation_station.io.completed.fire, reservation_station.io.completed.bits)
  }

  when (io.cmd.valid && io.cmd.bits.inst.funct === CLKGATE_EN && !io.busy) {
    clock_en_reg := io.cmd.bits.rs1(0)
  }

  val raw_cmd_q = Module(new Queue(new GemminiCmd(reservation_station_entries), entries = 5))
  raw_cmd_q.io.enq.valid := io.cmd.valid
  io.cmd.ready := raw_cmd_q.io.enq.ready
  raw_cmd_q.io.enq.bits.cmd := io.cmd.bits
  raw_cmd_q.io.enq.bits.rob_id := DontCare
  raw_cmd_q.io.enq.bits.from_conv_fsm := false.B
  raw_cmd_q.io.enq.bits.from_matmul_fsm := false.B

  val raw_cmd = raw_cmd_q.io.deq

  val max_lds = reservation_station_entries_ld
  val max_exs = reservation_station_entries_ex
  val max_sts = reservation_station_entries_st

  val (conv_cmd, loop_conv_unroller_busy, ext_loop_conv_ws) = withClock (gated_clock) { LoopConv(raw_cmd, reservation_station.io.conv_ld_completed, reservation_station.io.conv_st_completed, reservation_station.io.conv_ex_completed,
    meshRows*tileRows, coreMaxAddrBits, reservation_station_entries, max_lds, max_exs, max_sts, sp_banks * sp_bank_entries, acc_banks * acc_bank_entries,
    inputType.getWidth, accType.getWidth, dma_maxbytes,
    new ConfigMvinRs1(mvin_scale_t_bits, block_stride_bits, pixel_repeats_bits), new MvinRs2(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new ConfigMvoutRs2(acc_scale_t_bits, 32), new MvoutRs2(mvout_rows_bits, mvout_cols_bits, local_addr_t),
    new ConfigExRs1(acc_scale_t_bits), new PreloadRs(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new PreloadRs(mvout_rows_bits, mvout_cols_bits, local_addr_t),
    new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    has_training_convs, has_max_pool, has_first_layer_optimizations, has_dw_convs, use_shared_res_entries, nSharers) }

  val (loop_cmd, loop_matmul_unroller_busy, ext_loop_ws) = withClock (gated_clock) { LoopMatmul(conv_cmd, reservation_station.io.matmul_ld_completed, reservation_station.io.matmul_st_completed, reservation_station.io.matmul_ex_completed,
    meshRows*tileRows, coreMaxAddrBits, reservation_station_entries, max_lds, max_exs, max_sts, sp_banks * sp_bank_entries, acc_banks * acc_bank_entries,
    inputType.getWidth, accType.getWidth, dma_maxbytes, new MvinRs2(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new PreloadRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new PreloadRs(mvout_rows_bits, mvout_cols_bits, local_addr_t),
    new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new MvoutRs2(mvout_rows_bits, mvout_cols_bits, local_addr_t), use_shared_res_entries, use_vpu_fusion, nSharers) }

  val iterator_bitwidth = 16
  val concurrent_loops = 2
  val group_num = nSharers * concurrent_loops
  val group_w = if (use_vpu_fusion) 3 else log2Up(group_num)
  val use_group_control = use_shared_res_entries || use_vpu_fusion

  val ext_loop_ws_io = if (use_group_control) Some(IO(new LdBExIO(
    group_w, nSharers, iterator_bitwidth, use_vpu_fusion))) else None
  ext_loop_ws_io.foreach(_ <> ext_loop_ws.get)
  val ext_loop_conv_ws_io = if (use_shared_res_entries) Some(IO(new LdIExIO(group_w, nSharers, iterator_bitwidth))) else None
  ext_loop_conv_ws_io.foreach(_ <> ext_loop_conv_ws.get)

  // val unrolled_cmd = Queue(loop_cmd)
  loop_cmd.ready := false.B
  counters.io.event_io.connectEventSignal(CounterEvent.LOOP_MATMUL_ACTIVE_CYCLES, loop_matmul_unroller_busy)

  // Wire up controllers to ROB
  reservation_station.io.alloc.valid := false.B
  reservation_station.io.alloc.bits := loop_cmd.bits

  /*
  //-------------------------------------------------------------------------
  // finish muxing control signals to rob (risc) or tiler (cisc)
  when (raw_cmd.valid && is_cisc_funct && !rob.io.busy) {
    is_cisc_mode       := true.B
    raw_cisc_cmd.valid := true.B
    raw_cmd.ready      := raw_cisc_cmd.ready
  }
  .elsewhen (raw_cmd.valid && !is_cisc_funct && !tiler.io.busy) {
    is_cisc_mode       := false.B
    raw_risc_cmd.valid := true.B
    raw_cmd.ready      := raw_risc_cmd.ready
  }
  */

  //=========================================================================
  // Controllers
  //=========================================================================
  val load_controller = withClock (gated_clock) { Module(new LoadController(outer.config, coreMaxAddrBits, local_addr_t)) }
  val store_controller = withClock (gated_clock) { Module(new StoreController(outer.config, coreMaxAddrBits, local_addr_t)) }
  val ex_controller = withClock (gated_clock) { Module(new ExecuteController(xLen, tagWidth, outer.config)) }
  val vpu_d_controller = if (use_vpu_fusion) Some(withClock(gated_clock) {
    Module(new VpuDLoadController(outer.config))
  }) else None
  if (use_vpu_fusion) {
    val matrixReadArbiter = withClock(gated_clock) { Module(
      new GemminiVpuMatrixReadArbiter(vpuRowAddrBits.get, block_cols,
        accType.getWidth)) }
    matrixReadArbiter.io.execute <> ex_controller.io.vpuMatrixRead.get
    matrixReadArbiter.io.dload <> vpu_d_controller.get.io.matrixRead
    vpu_matrix_read_io.get <> matrixReadArbiter.io.out
    spad.module.io.acc.vpu_d_write.get <>
      vpu_d_controller.get.io.accWrite
  }

  counters.io.event_io.collect(load_controller.io.counter)
  counters.io.event_io.collect(store_controller.io.counter)
  counters.io.event_io.collect(ex_controller.io.counter)

  if (use_profiler) {
    val profiler = profilers.get
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ENTER_LD_CTRL, load_controller.io.cmd.fire, load_controller.io.cmd.bits.rob_id.bits)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ENTER_EX_CTRL, ex_controller.io.cmd.fire, ex_controller.io.cmd.bits.rob_id.bits)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.ENTER_ST_CTRL, store_controller.io.cmd.fire, store_controller.io.cmd.bits.rob_id.bits)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.LEAVE_LD_CTRL, load_controller.io.completed.fire, load_controller.io.completed.bits)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.LEAVE_EX_CTRL, ex_controller.io.completed.fire, ex_controller.io.completed.bits)
    profiler.module.io.profile_io.event_io.connectEventSignal(ProfileEvent.LEAVE_ST_CTRL, store_controller.io.completed.fire, store_controller.io.completed.bits)
    // profilers.io.profile_io.event_io.connectEventSignal(ProfileEvent.ENTER_DMA_READ, load_controller.io.dma.req.fire, load_controller.io.dma.req.bits.cmd_id)
    // profilers.io.profile_io.event_io.connectEventSignal(ProfileEvent.LEAVE_DMA_READ, load_controller.io.dma.resp.fire, load_controller.io.dma.req.bits.cmd_id)
    // profilers.io.profile_io.event_io.connectEventSignal(ProfileEvent.ENTER_DMA_WRITE, store_controller.io.dma.req.fire, store_controller.io.dma.req.bits.cmd_id)
    // profilers.io.profile_io.event_io.connectEventSignal(ProfileEvent.LEAVE_DMA_WRITE, store_controller.io.dma.resp.fire, store_controller.io.dma.resp.bits.cmd_id)
    profiler.module.io.profile_io.event_io.collect(load_controller.io.profile.get)
    profiler.module.io.profile_io.event_io.collect(ex_controller.io.profile.get)
    profiler.module.io.profile_io.event_io.collect(store_controller.io.profile.get)
  }

  /*
  tiler.io.issue.load.ready := false.B
  tiler.io.issue.store.ready := false.B
  tiler.io.issue.exec.ready := false.B
  */

  reservation_station.io.issue.ld.ready := false.B
  reservation_station.io.issue.st.ready := false.B
  reservation_station.io.issue.ex.ready := false.B

  /*
  when (is_cisc_mode) {
    load_controller.io.cmd  <> tiler.io.issue.load
    store_controller.io.cmd <> tiler.io.issue.store
    ex_controller.io.cmd  <> tiler.io.issue.exec
  }
  .otherwise {
    load_controller.io.cmd.valid := rob.io.issue.ld.valid
    rob.io.issue.ld.ready := load_controller.io.cmd.ready
    load_controller.io.cmd.bits.cmd := rob.io.issue.ld.cmd
    load_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.ld.cmd.inst.funct
    load_controller.io.cmd.bits.rob_id.push(rob.io.issue.ld.rob_id)

    store_controller.io.cmd.valid := rob.io.issue.st.valid
    rob.io.issue.st.ready := store_controller.io.cmd.ready
    store_controller.io.cmd.bits.cmd := rob.io.issue.st.cmd
    store_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.st.cmd.inst.funct
    store_controller.io.cmd.bits.rob_id.push(rob.io.issue.st.rob_id)

    ex_controller.io.cmd.valid := rob.io.issue.ex.valid
    rob.io.issue.ex.ready := ex_controller.io.cmd.ready
    ex_controller.io.cmd.bits.cmd := rob.io.issue.ex.cmd
    ex_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.ex.cmd.inst.funct
    ex_controller.io.cmd.bits.rob_id.push(rob.io.issue.ex.rob_id)
  }
  */

  load_controller.io.cmd.bits := reservation_station.io.issue.ld.cmd
  load_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.ld.rob_id)
  if (use_vpu_fusion) {
    val issuedLdLocalAddr = reservation_station.io.issue.ld.cmd.cmd.rs2(31, 0)
      .asTypeOf(local_addr_t)
    val issuedLdFromVsram =
      reservation_station.io.issue.ld.cmd.cmd.inst.funct === LOAD3_CMD &&
        issuedLdLocalAddr.is_acc_addr && issuedLdLocalAddr.d_from_vsram()

    load_controller.io.cmd.valid := reservation_station.io.issue.ld.valid &&
      !issuedLdFromVsram
    vpu_d_controller.get.io.cmd.valid :=
      reservation_station.io.issue.ld.valid && issuedLdFromVsram
    vpu_d_controller.get.io.cmd.bits := reservation_station.io.issue.ld.cmd
    vpu_d_controller.get.io.cmd.bits.rob_id.push(
      reservation_station.io.issue.ld.rob_id)
    reservation_station.io.issue.ld.ready := Mux(issuedLdFromVsram,
      vpu_d_controller.get.io.cmd.ready, load_controller.io.cmd.ready)
  } else {
    load_controller.io.cmd.valid := reservation_station.io.issue.ld.valid
    reservation_station.io.issue.ld.ready := load_controller.io.cmd.ready
  }

  store_controller.io.cmd.valid := reservation_station.io.issue.st.valid
  reservation_station.io.issue.st.ready := store_controller.io.cmd.ready
  store_controller.io.cmd.bits := reservation_station.io.issue.st.cmd
  store_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.st.rob_id)

  ex_controller.io.cmd.valid := reservation_station.io.issue.ex.valid
  reservation_station.io.issue.ex.ready := ex_controller.io.cmd.ready
  ex_controller.io.cmd.bits := reservation_station.io.issue.ex.cmd
  ex_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.ex.rob_id)

  // Wire up scratchpad to controllers
  spad.module.io.dma.read <> load_controller.io.dma
  spad.module.io.dma.write <> store_controller.io.dma
  ex_controller.io.srams.read <> spad.module.io.srams.read
  ex_controller.io.srams.write <> spad.module.io.srams.write
  spad.module.io.acc.read_req <> ex_controller.io.acc.read_req
  ex_controller.io.acc.read_resp <> spad.module.io.acc.read_resp
  ex_controller.io.acc.write <> spad.module.io.acc.write
  if (use_vpu_fusion) {
    spad.module.io.acc.write_to_vpu.get :=
      ex_controller.io.acc.write_to_vpu.get
  }
  if (use_shared_ext_mem) {
    ex_controller.io.srams.grant.get <> spad.module.io.exwrite_grant.get.spad
    ex_controller.io.acc.grant.get <> spad.module.io.exwrite_grant.get.acc
  }

  // Im2Col unit
  val im2col = withClock (gated_clock) { Module(new Im2Col(outer.config)) }

  // Wire up Im2col
  counters.io.event_io.collect(im2col.io.counter)
  // im2col.io.sram_reads <> spad.module.io.srams.read
  im2col.io.req <> ex_controller.io.im2col.req
  ex_controller.io.im2col.resp <> im2col.io.resp

  // Wire arbiter for ExecuteController and Im2Col scratchpad reads
  (ex_controller.io.srams.read, im2col.io.sram_reads, spad.module.io.srams.read).zipped.foreach { case (ex_read, im2col_read, spad_read) =>
    val req_arb = Module(new Arbiter(new ScratchpadReadReq(n=sp_bank_entries), 2))

    req_arb.io.in(0) <> ex_read.req
    req_arb.io.in(1) <> im2col_read.req

    spad_read.req <> req_arb.io.out

    // TODO if necessary, change how the responses are handled when fromIm2Col is added to spad read interface

    ex_read.resp.valid := spad_read.resp.valid
    im2col_read.resp.valid := spad_read.resp.valid

    ex_read.resp.bits := spad_read.resp.bits
    im2col_read.resp.bits := spad_read.resp.bits

    spad_read.resp.ready := ex_read.resp.ready || im2col_read.resp.ready
  }

  // Wire up controllers to ROB
  reservation_station.io.alloc.valid := false.B
  // rob.io.alloc.bits := compressed_cmd.bits
  reservation_station.io.alloc.bits := loop_cmd.bits

  /*
  //=========================================================================
  // committed insn return path to frontends
  //=========================================================================

  //-------------------------------------------------------------------------
  // cisc
  tiler.io.completed.exec.valid := ex_controller.io.completed.valid
  tiler.io.completed.exec.bits := ex_controller.io.completed.bits

  tiler.io.completed.load <> load_controller.io.completed
  tiler.io.completed.store <> store_controller.io.completed

  // mux with cisc frontend arbiter
  tiler.io.completed.exec.valid  := ex_controller.io.completed.valid && is_cisc_mode
  tiler.io.completed.load.valid  := load_controller.io.completed.valid && is_cisc_mode
  tiler.io.completed.store.valid := store_controller.io.completed.valid && is_cisc_mode
  */

  //-------------------------------------------------------------------------
  // risc
  val nCompletionSources = if (use_vpu_fusion) 4 else 3
  val reservation_station_completed_arb = Module(new Arbiter(
    UInt(log2Up(reservation_station_entries).W), nCompletionSources))

  reservation_station_completed_arb.io.in(0).valid := ex_controller.io.completed.valid
  reservation_station_completed_arb.io.in(0).bits := ex_controller.io.completed.bits

  reservation_station_completed_arb.io.in(1) <> load_controller.io.completed
  if (use_vpu_fusion) {
    reservation_station_completed_arb.io.in(2) <>
      vpu_d_controller.get.io.completed
  }
  val storeCompletionIndex = if (use_vpu_fusion) 3 else 2
  reservation_station_completed_arb.io.in(storeCompletionIndex) <>
    store_controller.io.completed

  // mux with cisc frontend arbiter
  reservation_station_completed_arb.io.in(0).valid := ex_controller.io.completed.valid // && !is_cisc_mode
  reservation_station_completed_arb.io.in(1).valid := load_controller.io.completed.valid // && !is_cisc_mode
  reservation_station_completed_arb.io.in(storeCompletionIndex).valid :=
    store_controller.io.completed.valid // && !is_cisc_mode

  reservation_station.io.completed.valid := reservation_station_completed_arb.io.out.valid
  reservation_station.io.completed.bits := reservation_station_completed_arb.io.out.bits
  reservation_station_completed_arb.io.out.ready := true.B

  // Wire up global RoCC signals
  val profiler_busy = if (use_profiler) profilers.get.module.io.busy else false.B
  io.busy := raw_cmd.valid || loop_conv_unroller_busy ||
    loop_matmul_unroller_busy || reservation_station.io.busy ||
    spad.module.io.busy ||
    vpu_d_controller.map(_.io.busy).getOrElse(false.B) || profiler_busy ||
    loop_cmd.valid || conv_cmd.valid

  io.interrupt := tlb.io.exp.map(_.interrupt).reduce(_ || _)

  // assert(!io.interrupt, "Interrupt handlers have not been written yet")

  // Cycle counters
  val incr_ld_cycles = load_controller.io.busy && !store_controller.io.busy && !ex_controller.io.busy
  val incr_st_cycles = !load_controller.io.busy && store_controller.io.busy && !ex_controller.io.busy
  val incr_ex_cycles = !load_controller.io.busy && !store_controller.io.busy && ex_controller.io.busy

  val incr_ld_st_cycles = load_controller.io.busy && store_controller.io.busy && !ex_controller.io.busy
  val incr_ld_ex_cycles = load_controller.io.busy && !store_controller.io.busy && ex_controller.io.busy
  val incr_st_ex_cycles = !load_controller.io.busy && store_controller.io.busy && ex_controller.io.busy

  val incr_ld_st_ex_cycles = load_controller.io.busy && store_controller.io.busy && ex_controller.io.busy

  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_CYCLES, incr_ld_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_ST_CYCLES, incr_st_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_EX_CYCLES, incr_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_ST_CYCLES, incr_ld_st_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_EX_CYCLES, incr_ld_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_ST_EX_CYCLES, incr_st_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_ST_EX_CYCLES, incr_ld_st_ex_cycles)

  // Issue commands to controllers
  // TODO we combinationally couple cmd.ready and cmd.valid signals here
  // when (compressed_cmd.valid) {
  when (loop_cmd.valid) {
    // val config_cmd_type = cmd.bits.rs1(1,0) // TODO magic numbers

    //val funct = unrolled_cmd.bits.inst.funct
    val risc_funct = loop_cmd.bits.cmd.inst.funct

    val is_flush = risc_funct === FLUSH_CMD
    val is_counter_op = risc_funct === COUNTER_OP
    val is_clock_gate_en = risc_funct === CLKGATE_EN
    val is_profiler_paddr = risc_funct === SET_PROFILER_PADDR

    /*
    val is_load = (funct === LOAD_CMD) || (funct === CONFIG_CMD && config_cmd_type === CONFIG_LOAD)
    val is_store = (funct === STORE_CMD) || (funct === CONFIG_CMD && config_cmd_type === CONFIG_STORE)
    val is_ex = (funct === COMPUTE_AND_FLIP_CMD || funct === COMPUTE_AND_STAY_CMD || funct === PRELOAD_CMD) ||
    (funct === CONFIG_CMD && config_cmd_type === CONFIG_EX)
    */

    when (is_flush) {
      val skip = loop_cmd.bits.cmd.rs1(0)
      tlb.io.exp.foreach(_.flush_skip := skip)
      tlb.io.exp.foreach(_.flush_retry := !skip)

      loop_cmd.ready := true.B // TODO should we wait for an acknowledgement from the TLB?
    }

    .elsewhen (is_counter_op) {
      // If this is a counter access/configuration command, execute immediately
      counters.io.in.valid := loop_cmd.valid
      loop_cmd.ready := counters.io.in.ready
      counters.io.in.bits := loop_cmd.bits.cmd
    }

    .elsewhen (is_clock_gate_en) {
      loop_cmd.ready := true.B
    }

    .elsewhen (is_profiler_paddr){
      if (use_profiler) {
        profilers.get.module.io.profiler_vaddr_valid := loop_cmd.valid
        profilers.get.module.io.profiler_vaddr := loop_cmd.bits.cmd.rs1
        profilers.get.module.io.profiler_status := loop_cmd.bits.cmd.status
      }
      loop_cmd.ready := true.B
    }

    .otherwise {
      reservation_station.io.alloc.valid := true.B

      when(reservation_station.io.alloc.fire) {
        // compressed_cmd.ready := true.B
        loop_cmd.ready := true.B
      }
    }
  }

  // Debugging signals
  val pipeline_stall_counter = RegInit(0.U(32.W))
  when (io.cmd.fire()) {
    pipeline_stall_counter := 0.U
  }.elsewhen(io.busy) {
    pipeline_stall_counter := pipeline_stall_counter + 1.U
  }
  assert(pipeline_stall_counter < 10000000.U, "pipeline stall")

  /*
  //=========================================================================
  // Wire up global RoCC signals
  //=========================================================================
  io.busy := raw_cmd.valid || unrolled_cmd.valid || rob.io.busy || spad.module.io.busy || tiler.io.busy
  io.interrupt := tlb.io.exp.interrupt

  // hack
  when(is_cisc_mode || !(unrolled_cmd.valid || rob.io.busy || tiler.io.busy)){
    tlb.io.exp.flush_retry := cmd_fsm.io.flush_retry
    tlb.io.exp.flush_skip  := cmd_fsm.io.flush_skip
  }
  */

  //=========================================================================
  // Performance Counters Access
  //=========================================================================

}
