package gemmini

import chisel3._
import chisel3.util._

object MakeWireBundle {
  def apply[T <: Bundle](gen: T, exp: T => (Data, Data)*): T = {
    val ret = Wire(gen)
    exp.foreach { e =>
      val (x, y) = e(ret)
      x := y
    }
    ret
  }
}

object MakeValid {
  def apply[T <: Data](valid: Bool, bits: T): ValidIO[T] = {
    // Explicitly stating the type here allows the function types to be
    // inferred.
    MakeWireBundle[ValidIO[T]](
      Valid(chiselTypeOf(bits)),
      _.valid -> valid,
      _.bits -> bits,
    )
  }

  def apply[T <: Data](bits: T): ValidIO[T] = {
    apply(true.B, bits)
  }
}

object MakeInvalid {
  def apply[T <: Data](gen: T): ValidIO[T] = MakeWireBundle[ValidIO[T]](
    Valid(gen),
    _.valid -> false.B,
    _.bits -> 0.U.asTypeOf(gen),
  )
}

object RotateVectorLeft {
  def apply[T <: Data](data: Vec[T], shift: UInt): Vec[T] = {
    val elemSize = data(0).asUInt.getWidth
    val rotated = data.asUInt.rotateLeft(shift * elemSize.U)
    rotated.asTypeOf(chiselTypeOf(data))
  }
}

// Rotates elements in a vector.
// Result[i] = data[(i - shift) % data.size]
object RotateVectorRight {
  def apply[T <: Data](data: Vec[T], shift: UInt): Vec[T] = {
    val elemSize = data(0).asUInt.getWidth
    val rotated = data.asUInt.rotateRight(shift * elemSize.U)
    rotated.asTypeOf(chiselTypeOf(data))
  }
}

// from coral npu
class CircularBuffer[T <: Data](t: T, nSharers: Int, capacity: Int) extends Module {
  // For the time being, restrict to powers of 2
  assert(isPow2(nSharers))
  assert(isPow2(capacity))
  val io = IO(new Bundle {
    val enqValid = Input(UInt(log2Ceil(nSharers + 1).W))
    val enqData = Input(Vec(nSharers, t))

    val nEnqueued = Output(UInt(log2Ceil(capacity + 1).W))
    val nSpace = Output(UInt(log2Ceil(capacity + 1).W))

    val dataOut = Output(t)
    val deqReady = Input(Bool())
    val deqValid = Output(Bool())

    def deqFire(dummy: Int = 0): Bool = deqReady && deqValid

    val flush = Input(Bool())
  })
  dontTouch(io)

  // Note first assert below should be sufficient it allows enqueueing items when buffer
  // is full or close to full provided deqReady >= enqValid.
  // The second assert is more conservative, and will never allow enqueueing more items
  // than there is space in the buffer, under any circumstances. May be removed if
  // desire for more is greater than the need to be more conservative.
  // assert(io.nEnqueued +& io.enqValid -& io.deqFire() <= capacity.U)
  // assert(io.enqValid <= (capacity.U -& io.nEnqueued))

//   assert(io.deqReady <= io.nEnqueued)

  val buffer = RegInit(VecInit.fill(capacity)(0.U.asTypeOf(t)))
  val enqPtr = RegInit(0.U(log2Ceil(capacity).W))
  val deqPtr = RegInit(0.U(log2Ceil(capacity).W))

  val expandedInput = Wire(Vec(capacity, Valid(t)))
  for (i <- 0 until capacity) {
    if (i < nSharers) {
      expandedInput(i) := MakeValid(i.U < io.enqValid, io.enqData(i))
    } else {
      expandedInput(i) := MakeInvalid(t)
    }
  }

  val rotatedInput = RotateVectorLeft(expandedInput, enqPtr)
  for (i <- 0 until capacity) {
    buffer(i) := Mux(rotatedInput(i).valid, rotatedInput(i).bits, buffer(i))
  }

  var nEnqueued = RegInit(0.U(io.nEnqueued.getWidth.W))
  enqPtr    := Mux(io.flush, 0.U, enqPtr + io.enqValid)
  deqPtr    := Mux(io.flush, 0.U, deqPtr + io.deqFire())
  nEnqueued := Mux(io.flush, 0.U, nEnqueued + io.enqValid - io.deqFire())

  io.nEnqueued := nEnqueued
  io.nSpace := capacity.U - nEnqueued
  io.deqValid := nEnqueued > 0.U

  val outputBufferView = RotateVectorRight(buffer, deqPtr)
  io.dataOut := outputBufferView(0)
}