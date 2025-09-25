package gemmini

import chisel3._
import chisel3.util._
import Util._

class SharedExtEntries(nSharers: Int, local_addr_t: LocalAddr, reservation_station_entries_ld: Int, reservation_station_entries_ex: Int, reservation_station_entries_st: Int, res_max_per_type: Int) extends Module {

    val io = IO(new Bundle {
        val in = Vec(nSharers, Flipped(new EntriesForDeps(local_addr_t, reservation_station_entries_ld, reservation_station_entries_ex, reservation_station_entries_st, res_max_per_type)))
    })

    val ldq :: exq :: stq :: Nil = Enum(3)

    class Entry(reservation_station_entries_0: Int, reservation_station_entries_1: Int) extends Bundle {
        val opa = UDValid(new OpT(local_addr_t))
        val opb = UDValid(new OpT(local_addr_t))
        val opa_is_dst = Bool()

        val deps_0 = Vec(reservation_station_entries_0*nSharers, Bool())
        val deps_1 = Vec(reservation_station_entries_1*nSharers, Bool())

        def ready(dummy: Int = 0): Bool = !(deps_0.reduce(_ || _) || deps_1.reduce(_ || _))
    }

    val total_entries_ld = Reg(Vec(reservation_station_entries_ld*nSharers, UDValid(new Entry(reservation_station_entries_ex, reservation_station_entries_st))))
    val total_entries_ex = Reg(Vec(reservation_station_entries_ex*nSharers, UDValid(new Entry(reservation_station_entries_ld, reservation_station_entries_st))))
    val total_entries_st = Reg(Vec(reservation_station_entries_st*nSharers, UDValid(new Entry(reservation_station_entries_ld, reservation_station_entries_ex))))


    for (i <- 0 until nSharers) {
        when(io.in(i).alloc_entry.valid) {
            val type_width = log2Up(res_max_per_type)
            val queue_type = io.in(i).alloc_entry.bits.alloc_id(type_width + 1, type_width)
            val issue_id = io.in(i).alloc_entry.bits.alloc_id(type_width - 1, 0)

            Seq((ldq, total_entries_ld, reservation_station_entries_ld),
                (exq, total_entries_ex, reservation_station_entries_ex),
                (stq, total_entries_st, reservation_station_entries_st))
                .foreach { case (q, total_entry_type, reservation_station_entries) =>
                    when (q === queue_type) {
                        val global_issue_id = (i*reservation_station_entries).U +& issue_id

                        val q_Int: Int = q.litValue.toInt

                        if (q_Int == 0) {
                            // war (after ex/st) | waw (after ex)
                            total_entry_type(global_issue_id).bits.deps_0 := VecInit(total_entries_ex.map { e => e.valid && io.in(i).alloc_entry.bits.not_config && (
                                (io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits) && e.bits.opa.valid) || // waw if preload, war if compute
                                (io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opb.bits) && e.bits.opb.valid))}) // war

                            total_entry_type(global_issue_id).bits.deps_1 := VecInit(total_entries_st.map { e => e.valid && e.bits.opa.valid && io.in(i).alloc_entry.bits.not_config &&
                                io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits)})  // war
                        } else if (q_Int == 1) {
                            // raw (after ld) | war (after st) | waw (after ld)
                            total_entry_type(global_issue_id).bits.deps_0 := VecInit(total_entries_ld.map { e => e.valid && e.bits.opa.valid && io.in(i).alloc_entry.bits.not_config && (
                                io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits) || // waw if preload, raw if compute
                                io.in(i).alloc_entry.bits.opb.bits.overlaps(e.bits.opa.bits))}) // raw

                            total_entry_type(global_issue_id).bits.deps_1 := VecInit(total_entries_st.map { e => e.valid && e.bits.opa.valid && io.in(i).alloc_entry.bits.not_config &&
                                io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits)})  // war
                        } else if (q_Int == 2) {
                            // raw (after ld/ex)
                            total_entry_type(global_issue_id).bits.deps_0 := VecInit(total_entries_ld.map { e => e.valid && e.bits.opa.valid && io.in(i).alloc_entry.bits.not_config &&
                                io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits)})  // raw

                            total_entry_type(global_issue_id).bits.deps_1 := VecInit(total_entries_ex.map { e => e.valid && e.bits.opa.valid && io.in(i).alloc_entry.bits.not_config &&
                                e.bits.opa_is_dst && io.in(i).alloc_entry.bits.opa.bits.overlaps(e.bits.opa.bits)}) // raw only if ex is preload
                        }

                        total_entry_type(global_issue_id).valid := true.B
                        total_entry_type(global_issue_id).bits.opa := io.in(i).alloc_entry.bits.opa
                        total_entry_type(global_issue_id).bits.opb := io.in(i).alloc_entry.bits.opb
                        total_entry_type(global_issue_id).bits.opa_is_dst := io.in(i).alloc_entry.bits.opa_is_dst
                    }
                }
        }
    }

    for (i <- 0 until nSharers) {
        val ld_deps_slice = total_entries_ld.slice(i*reservation_station_entries_ld, (i+1)*reservation_station_entries_ld)
        io.in(i).ld_deps_ready := ld_deps_slice.map(e => e.bits.ready())
        val ex_deps_slice = total_entries_ex.slice(i*reservation_station_entries_ex, (i+1)*reservation_station_entries_ex)
        io.in(i).ex_deps_ready := ex_deps_slice.map(e => e.bits.ready())
        val st_deps_slice = total_entries_st.slice(i*reservation_station_entries_st, (i+1)*reservation_station_entries_st)
        io.in(i).st_deps_ready := st_deps_slice.map(e => e.bits.ready())
    }

    for (i <- 0 until nSharers) {
        when(io.in(i).issue_ld.valid) {
            val issue_id = io.in(i).issue_ld.bits.issue_id
            val global_issue_id = (i*reservation_station_entries_ld).U +& issue_id

            total_entries_ld(global_issue_id).valid := io.in(i).issue_ld.bits.valid
            when (!io.in(i).issue_ld.bits.valid) {
                total_entries_ex.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_0(global_issue_id) := false.B
                }
                total_entries_st.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_0(global_issue_id) := false.B
                }
            }
        }
        when(io.in(i).issue_ex.valid) {
            val issue_id = io.in(i).issue_ex.bits.issue_id
            val global_issue_id = (i*reservation_station_entries_ex).U +& issue_id

            total_entries_ex(global_issue_id).valid := io.in(i).issue_ex.bits.valid
            when (!io.in(i).issue_ex.bits.valid) {
                total_entries_ld.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_0(global_issue_id) := false.B
                }
                total_entries_st.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_1(global_issue_id) := false.B
                }
            }
        }
        when(io.in(i).issue_st.valid) {
            val issue_id = io.in(i).issue_st.bits.issue_id
            val global_issue_id = (i*reservation_station_entries_st).U +& issue_id

            total_entries_st(global_issue_id).valid := io.in(i).issue_st.bits.valid
            when (!io.in(i).issue_st.bits.valid) {
                total_entries_ld.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_1(global_issue_id) := false.B
                }
                total_entries_ex.zipWithIndex.foreach { case (e, k) =>
                    e.bits.deps_1(global_issue_id) := false.B
                }
            }
        }
    }

    for (i <- 0 until nSharers) {
        when(io.in(i).complete_id.valid) {
            val type_width = log2Up(res_max_per_type)
            val queue_type = io.in(i).complete_id.bits(type_width + 1, type_width)
            val issue_id = io.in(i).complete_id.bits(type_width - 1, 0)

            when (queue_type === ldq) {
                val global_issue_id = (i*reservation_station_entries_ld).U +& issue_id
                total_entries_ex.foreach(_.bits.deps_0(global_issue_id) := false.B)
                total_entries_st.foreach(_.bits.deps_0(global_issue_id) := false.B)

                total_entries_ld(global_issue_id).valid := false.B
            }.elsewhen (queue_type === exq) {
                val global_issue_id = (i*reservation_station_entries_ex).U +& issue_id
                total_entries_ld.foreach(_.bits.deps_0(global_issue_id) := false.B)
                total_entries_st.foreach(_.bits.deps_1(global_issue_id) := false.B)

                total_entries_ex(global_issue_id).valid := false.B
            }.elsewhen (queue_type === stq) {
                val global_issue_id = (i*reservation_station_entries_st).U +& issue_id
                total_entries_ld.foreach(_.bits.deps_1(global_issue_id) := false.B)
                total_entries_ex.foreach(_.bits.deps_1(global_issue_id) := false.B)

                total_entries_st(global_issue_id).valid := false.B
            }.otherwise {
                assert(queue_type =/= 3.U)
            }
        }
    }

    // Explicitly mark "opb" in all ld/st queues entries as being invalid.
    // This helps us to reduce the total reservation table area
    Seq(total_entries_ld, total_entries_st).foreach { entries_type =>
        entries_type.foreach { e =>
            e.bits.opb.valid := false.B
            e.bits.opb.bits := DontCare
        }
    }

    when (reset.asBool) {
        total_entries_ld.foreach(_.valid := false.B)
        total_entries_ex.foreach(_.valid := false.B)
        total_entries_st.foreach(_.valid := false.B)
    }
}