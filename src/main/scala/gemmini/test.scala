package chisel3.util

import chisel3._
// import chisel3.util.HasBlackBoxInline
import chisel3.internal.{Builder, NamedComponent}
import chisel3.experimental.{OpaqueType}
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
import chisel3.internal.sourceinfo.MemTransform
import chisel3.util.experimental.loadMemoryFromFileInline
import firrtl.annotations.{IsMember, MemoryLoadFileType}

import scala.language.reflectiveCalls

import scala.collection.immutable.{ListMap, VectorMap}

/** A bundle of signals representing a memory read port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  */
class MemoryReadPort[T <: Data](tpe: T, addrWidth: Int) extends Bundle {
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val data = Output(tpe)
}

/** A bundle of signals representing a memory write port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  * @param masked Whether this read/write port should have an optional mask.
  *
  * @note `masked` is only valid if tpe is a Vec; if this is not the case no mask will be initialized
  *       regardless of the value of `masked`.
  */
class MemoryWritePort[T <: Data](tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val data = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
}

/** A bundle of signals representing a memory read/write port.
  *
  * @tparam tpe The data type of the memory port
  * @param addrWidth The width of the address signal
  * @param masked Whether this read/write port should have an optional mask.
  *
  * @note `masked` is only valid if tpe is a Vec; if this is not the case no mask will be initialized
  *       regardless of the value of `masked`.
  */
class MemoryReadWritePort[T <: Data](tpe: T, addrWidth: Int, masked: Boolean) extends Bundle {
  val address = Input(UInt(addrWidth.W))
  val enable = Input(Bool())
  val isWrite = Input(Bool())
  val readData = Output(tpe)
  val writeData = Input(tpe)
  val mask: Option[Vec[Bool]] = if (masked) {
    val maskSize = tpe match {
      case vec: Vec[_] => vec.size
      case _ => 0
    }
    Some(Input(Vec(maskSize, Bool())))
  } else {
    None
  }
}

/** A IO bundle of signals connecting to the ports of a memory, as requested by
  * `SRAMInterface.apply`.
  *
  * @param memSize The size of the memory, used to calculate the address width
  * @tparam tpe The data type of the memory port
  * @param numReadPorts The number of read ports
  * @param numWritePorts The number of write ports
  * @param numReadwritePorts The number of read/write ports
  * @param masked Whether the memory is write masked
  * @param hasDescription Whether this interface contains an [[SRAMDescription]]
  */
class SRAMInterface[T <: Data](
  val memSize: BigInt,
  // tpe can't be directly made public as it will become a Bundle field
  tpe:                   T,
  val numReadPorts:      Int,
  val numWritePorts:     Int,
  val numReadwritePorts: Int,
  val masked:            Boolean = false
) extends Bundle {

  /** Public accessor for data type of this interface. */
  def dataType: T = tpe

  if (masked) {
    require(
      tpe.isInstanceOf[Vec[_]],
      s"masked writes require that SRAMInterface is instantiated with a data type of Vec (got $tpe instead)"
    )
  }
  // override def typeName: String =
  //   s"SRAMInterface_${SRAM.portedness(numReadPorts, numWritePorts, numReadwritePorts)}${if (masked) "_masked"
  //     else ""}}_${tpe.typeName}"

  val addrWidth = log2Up(memSize)

  val readPorts:  Vec[MemoryReadPort[T]] = Vec(numReadPorts, new MemoryReadPort(tpe, addrWidth))
  val writePorts: Vec[MemoryWritePort[T]] = Vec(numWritePorts, new MemoryWritePort(tpe, addrWidth, masked))
  val readwritePorts: Vec[MemoryReadWritePort[T]] =
    Vec(numReadwritePorts, new MemoryReadWritePort(tpe, addrWidth, masked))
}

class SRAMBlackbox(numReadPorts: Int, numWritePorts: Int, numReadwritePorts: Int, depth: Int, masked: Boolean, mask_len: Int, data_len: Int) extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle{
    val RW0_addr  = Input(UInt(log2Ceil(depth).W))
    val RW1_addr  = Input(UInt(log2Ceil(depth).W))
    val RW2_addr  = Input(UInt(log2Ceil(depth).W))
    val RW3_addr  = Input(UInt(log2Ceil(depth).W))
    val RW0_en    = Input(Bool())
    val RW1_en    = Input(Bool())
    val RW2_en    = Input(Bool())
    val RW3_en    = Input(Bool())
    val RW0_clk   = Input(Clock())
    val RW1_clk   = Input(Clock())
    val RW2_clk   = Input(Clock())
    val RW3_clk   = Input(Clock())
    val RW0_wmode = Input(Bool())
    val RW1_wmode = Input(Bool())
    val RW2_wmode = Input(Bool())
    val RW3_wmode = Input(Bool())
    val RW0_wdata = Input(UInt((data_len * mask_len).W))
    val RW1_wdata = Input(UInt((data_len * mask_len).W))
    val RW2_wdata = Input(UInt((data_len * mask_len).W))
    val RW3_wdata = Input(UInt((data_len * mask_len).W))
    val RW0_rdata = Output(UInt((data_len * mask_len).W))
    val RW1_rdata = Output(UInt((data_len * mask_len).W))
    val RW2_rdata = Output(UInt((data_len * mask_len).W))
    val RW3_rdata = Output(UInt((data_len * mask_len).W))
    val RW0_wmask = if (masked) Some(Input(UInt(mask_len.W))) else None
    val RW1_wmask = if (masked) Some(Input(UInt(mask_len.W))) else None
    val RW2_wmask = if (masked) Some(Input(UInt(mask_len.W))) else None
    val RW3_wmask = if (masked) Some(Input(UInt(mask_len.W))) else None
  })

  private val verilogInterface: String =
    (Seq.tabulate(numWritePorts)(idx =>
      Seq(
        s"// Write Port $idx",
        s"input [${log2Ceil(depth) - 1}:0] W${idx}_addr",
        s"input W${idx}_en",
        s"input W${idx}_clk",
        s"input [${data_len * mask_len - 1}:0] W${idx}_data"
      ) ++
        Option.when(masked)(s"input [${mask_len - 1}:0] W${idx}_mask")
    ) ++
      Seq.tabulate(numReadPorts)(idx =>
        Seq(
          s"// Read Port $idx",
          s"input [${log2Ceil(depth) - 1}:0] R${idx}_addr",
          s"input R${idx}_en",
          s"input R${idx}_clk",
          s"output [${data_len * mask_len - 1}:0] R${idx}_data"
        )
      ) ++
      Seq.tabulate(numReadwritePorts)(idx =>
        Seq(
          s"// ReadWrite Port $idx",
          s"input [${log2Ceil(depth) - 1}:0] RW${idx}_addr",
          s"input RW${idx}_en",
          s"input RW${idx}_clk",
          s"input RW${idx}_wmode",
          s"input [${data_len * mask_len - 1}:0] RW${idx}_wdata",
          s"output [${data_len * mask_len - 1}:0] RW${idx}_rdata"
        ) ++ Option
          .when(masked)(
            s"input [${mask_len - 1}:0] RW${idx}_wmask"
          )
      )).flatten.mkString(",\n")

  private val rLogic = Seq
    .tabulate(numReadPorts) { idx =>
      val prefix = s"R${idx}"
      Seq(
        s"reg _${prefix}_en;",
        s"reg [${log2Ceil(depth) - 1}:0] _${prefix}_addr;"
      ) ++
        Seq(
          s"always @(posedge ${prefix}_clk) begin // ${prefix}",
          s"_${prefix}_en <= ${prefix}_en;",
          s"_${prefix}_addr <= ${prefix}_addr;",
          s"end // ${prefix}"
        ) ++
        Some(s"assign ${prefix}_data = _${prefix}_en ? Memory[_${prefix}_addr] : ${data_len * mask_len}'bx;")
    }
    .flatten

  private val wLogic = Seq
    .tabulate(numWritePorts) { idx =>
      val prefix = s"W${idx}"
      Seq(s"always @(posedge ${prefix}_clk) begin // ${prefix}") ++
        (if (masked)
           Seq.tabulate(mask_len)(i =>
             s"if (${prefix}_en & ${prefix}_mask[${i}]) Memory[${prefix}_addr][${i * data_len} +: ${data_len}] <= ${prefix}_data[${(i + 1) * data_len - 1}:${i * data_len}];"
           )
         else
           Seq(s"if (${prefix}_en) Memory[${prefix}_addr] <= ${prefix}_data;")) ++
        Seq(s"end // ${prefix}")
    }
    .flatten

  private val rwLogic = Seq
    .tabulate(numReadwritePorts) { idx =>
      val prefix = s"RW${idx}"
      Seq(
        s"reg [${log2Ceil(depth) - 1}:0] _${prefix}_raddr;",
        s"reg _${prefix}_ren;",
        s"reg _${prefix}_rmode;"
      ) ++
        Seq(s"always @(posedge ${prefix}_clk) begin // ${prefix}") ++
        Seq(
          s"_${prefix}_raddr <= ${prefix}_addr;",
          s"_${prefix}_ren <= ${prefix}_en;",
          s"_${prefix}_rmode <= ${prefix}_wmode;"
        ) ++
        (if (masked)
           Seq.tabulate(mask_len)(i =>
             s"if(${prefix}_en & ${prefix}_wmask[${i}] & ${prefix}_wmode) Memory[${prefix}_addr][${i * data_len} +: ${data_len}] <= ${prefix}_wdata[${(i + 1) * data_len - 1}:${i * data_len}];"
           )
         else
           Seq(s"if (${prefix}_en & ${prefix}_wmode) Memory[${prefix}_addr] <= ${prefix}_wdata;")) ++
        Seq(s"end // ${prefix}") ++
        Seq(
          s"assign ${prefix}_rdata = _${prefix}_ren & ~_${prefix}_rmode ? Memory[_${prefix}_raddr] : ${data_len * mask_len}'bx;"
        )
    }
    .flatten

  private val logic =
    (Seq(s"reg [${data_len * mask_len - 1}:0] Memory[0:${depth - 1}];") ++ wLogic ++ rLogic ++ rwLogic)
      .mkString("\n")

  override def desiredName: String = s"BlackBoxSRAM_d${depth}_m${mask_len}_w${data_len}"

  setInline(
    desiredName + ".sv",
    s"""module ${desiredName}(
       |${verilogInterface}
       |);
       |${logic}
       |endmodule
       |""".stripMargin
  )
}

object SRAM {
  def apply[T <: Data](
      size:              BigInt,
      tpe:               T,
      numReadPorts:      Int,
      numWritePorts:     Int,
      numReadwritePorts: Int,
      masked:            Boolean = false
  ): SRAMInterface[T] = {
    val addrWidth = log2Ceil(size)
    val mask_len = tpe match {
      case vec: Vec[_] => vec.length
      case _           => 1
    }
    val data_len = tpe match {
      case vec: Vec[_] => vec.head.getWidth
      case dt          => dt.getWidth
    }

    val sram = Module(new SRAMBlackbox(numReadPorts, numWritePorts, numReadwritePorts, size.toInt, masked, mask_len, data_len))

    val io = Wire(new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, masked))

    for (i <- 0 until numReadPorts) {
      sram.suggestName(s"sram_read_$i")
      sram.io.elements(s"R${i}_addr") := io.readPorts(i).address
      sram.io.elements(s"R${i}_en") := io.readPorts(i).enable
      sram.io.elements(s"R${i}_clk") := Builder.forcedClock
      io.readPorts(i).data := sram.io.elements(s"R${i}_data").asTypeOf(tpe)
    }

    for (i <- 0 until numWritePorts) {
      sram.io.elements(s"W${i}_addr") := io.writePorts(i).address
      sram.io.elements(s"W${i}_en") := io.writePorts(i).enable
      sram.io.elements(s"W${i}_clk") := Builder.forcedClock
      sram.io.elements(s"W${i}_data") := io.writePorts(i).data.asUInt
      if (masked) {
        sram.io.elements(s"W${i}_mask").asInstanceOf[UInt] := io.writePorts(i).mask.get.asUInt
      }
    }

    for (i <- 0 until numReadwritePorts) {
      sram.io.elements(s"RW${i}_addr") := io.readwritePorts(i).address
      sram.io.elements(s"RW${i}_en") := io.readwritePorts(i).enable
      sram.io.elements(s"RW${i}_clk") := Builder.forcedClock
      sram.io.elements(s"RW${i}_wmode") := io.readwritePorts(i).isWrite
      sram.io.elements(s"RW${i}_wdata") := io.readwritePorts(i).writeData.asUInt
      io.readwritePorts(i).readData := sram.io.elements(s"RW${i}_rdata").asTypeOf(tpe)
      if (masked) {
        sram.io.elements(s"RW${i}_wmask").asInstanceOf[UInt] := io.readwritePorts(i).mask.get.asUInt
      }
    }

    io
  }
}

// object SRAM {

//   /** Generates a memory within the current module, connected to an explicit number
//     * of read, write, and read/write ports. This SRAM abstraction has both read and write capabilities: that is,
//     * it contains at least one read accessor (a read-only or read-write port), and at least one write accessor
//     * (a write-only or read-write port).
//     *
//     * @param size The desired size of the inner `SyncReadMem`
//     * @tparam T The data type of the memory element
//     * @param numReadPorts The number of desired read ports >= 0, and (numReadPorts + numReadwritePorts) > 0
//     * @param numWritePorts The number of desired write ports >= 0, and (numWritePorts + numReadwritePorts) > 0
//     * @param numReadwritePorts The number of desired read/write ports >= 0, and the above two conditions must hold
//     *
//     * @return A new `SRAMInterface` wire containing the control signals for each instantiated port
//     * @note This does *not* return the `SyncReadMem` itself, you must interact with it using the returned bundle
//     * @note Read-only memories (R >= 1, W === 0, RW === 0) and write-only memories (R === 0, W >= 1, RW === 0) are not supported by this API, and will result in an error if declared.
//     */
//   def apply[T <: Data](
//     size:              BigInt,
//     tpe:               T,
//     numReadPorts:      Int,
//     numWritePorts:     Int,
//     numReadwritePorts: Int,
//   ): SRAMInterface[T] = {
//     val clock = Builder.forcedClock
//     memInterface_impl(
//       size,
//       tpe,
//       Seq.fill(numReadPorts)(clock),
//       Seq.fill(numWritePorts)(clock),
//       Seq.fill(numReadwritePorts)(clock)
//     )
//   }

//   private def memInterface_blackbox_impl[T <: Data](
//     size:                BigInt,
//     tpe:                 T,
//     readPortClocks:      Seq[Clock],
//     writePortClocks:     Seq[Clock],
//     readwritePortClocks: Seq[Clock]
//   ): SRAMInterface[T] = {
//     val numReadPorts = readPortClocks.size
//     val numWritePorts = writePortClocks.size
//     val numReadwritePorts = readwritePortClocks.size
//     val enableMask = tpe match {
//       case vec: Vec[_] => true
//       case dt          => false
//     }

//     val isValidSRAM = ((numReadPorts + numReadwritePorts) > 0) && ((numWritePorts + numReadwritePorts) > 0)
//     val maskGranularity = tpe match {
//       case vec: Vec[_] if enableMask => vec.sample_element.getWidth
//       case _ => 0
//     }
//     val mask_len = tpe match {
//       case vec: Vec[_] => vec.length
//       case _           => 1
//     }
//     val data_len = tpe match {
//       case vec: Vec[_] => vec.sample_element.getWidth
//       case dt          => dt.getWidth
//     }

//     if (!isValidSRAM) {
//       val badMemory =
//         if (numReadPorts + numReadwritePorts == 0)
//           "write-only SRAM (R + RW === 0)"
//         else
//           "read-only SRAM (W + RW === 0)"
//       Builder.error(
//         s"Attempted to initialize a $badMemory! SRAMs must have both at least one read accessor and at least one write accessor."
//       )
//     }

//     // val mem = Instantiate(
//     val mem = Instantiate(
//       new SRAMBlackbox(
//           numReadPorts,
//           numWritePorts,
//           numReadwritePorts,
//           size.intValue,
//           enableMask,
//           mask_len,
//           data_len
//       )
//     )

//     implicit class SRAMInstanceMethods(underlying: Instance[SRAMBlackbox]) {
//       implicit val mg: internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
//       def io = underlying._lookup(_.io)
//     }

//     val sramReadPorts = Seq.tabulate(numReadPorts)(i => mem.io.R(i))
//     val sramWritePorts = Seq.tabulate(numWritePorts)(i => mem.io.W(i))
//     val sramReadwritePorts = Seq.tabulate(numReadwritePorts)(i => mem.io.RW(i))

//     val out = Wire(
//       new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, enableMask)
//     )

//     out.readPorts.zip(sramReadPorts).zip(readPortClocks).map { case ((intfReadPort, sramReadPort), readClock) =>
//       sramReadPort.address := intfReadPort.address
//       sramReadPort.clock := readClock
//       intfReadPort.data := sramReadPort.data.asTypeOf(tpe)
//       sramReadPort.enable := intfReadPort.enable
//     }
//     out.writePorts.zip(sramWritePorts).zip(writePortClocks).map { case ((intfWritePort, sramWritePort), writeClock) =>
//       sramWritePort.address := intfWritePort.address
//       sramWritePort.clock := writeClock
//       sramWritePort.data := intfWritePort.data.asUInt
//       sramWritePort.enable := intfWritePort.enable
//       sramWritePort.mask match {
//         case Some(mask) => mask := intfWritePort.mask.get.asUInt
//         case None       => assert(intfWritePort.mask.isEmpty)
//       }
//     }
//     out.readwritePorts.zip(sramReadwritePorts).zip(readwritePortClocks).map {
//       case ((intfReadwritePort, sramReadwritePort), readwriteClock) =>
//         sramReadwritePort.address := intfReadwritePort.address
//         sramReadwritePort.clock := readwriteClock
//         sramReadwritePort.enable := intfReadwritePort.enable
//         intfReadwritePort.readData := sramReadwritePort.readData.asTypeOf(tpe)
//         sramReadwritePort.writeData := intfReadwritePort.writeData.asUInt
//         sramReadwritePort.writeEnable := intfReadwritePort.isWrite
//         sramReadwritePort.writeMask match {
//           case Some(mask) => mask := intfReadwritePort.mask.get.asUInt
//           case None       => assert(intfReadwritePort.mask.isEmpty)
//         }
//     }
//     out
//   }

//   private def memInterface_impl[T <: Data](
//     size:                BigInt,
//     tpe:                 T,
//     readPortClocks:      Seq[Clock],
//     writePortClocks:     Seq[Clock],
//     readwritePortClocks: Seq[Clock]
//   ): SRAMInterface[T] = {
//     memInterface_blackbox_impl(
//       size,
//       tpe,
//       readPortClocks,
//       writePortClocks,
//       readwritePortClocks
//     )

//     val numReadPorts = readPortClocks.size
//     val numWritePorts = writePortClocks.size
//     val numReadwritePorts = readwritePortClocks.size
//     val isVecMem = tpe match {
//       case vec: Vec[_] => true
//       case dt          => false
//     }
//     val isValidSRAM = ((numReadPorts + numReadwritePorts) > 0) && ((numWritePorts + numReadwritePorts) > 0)

//     if (!isValidSRAM) {
//       val badMemory =
//         if (numReadPorts + numReadwritePorts == 0)
//           "write-only SRAM (R + RW === 0)"
//         else
//           "read-only SRAM (W + RW === 0)"
//       Builder.error(
//         s"Attempted to initialize a $badMemory! SRAMs must have both at least one read accessor and at least one write accessor."
//       )
//     }

//     // underlying target
//     val mem = withName("sram")(new SramTarget)

//     // user-facing interface into the SRAM
//     val sramIntfType =
//       new SRAMInterface(size, tpe, numReadPorts, numWritePorts, numReadwritePorts, isVecMem)
//     val _out = Wire(sramIntfType)
//     _out._underlying = Some(HasTarget(mem))

//     // create actual ports into firrtl memory
//     val firrtlReadPorts:  Seq[FirrtlMemoryReader[_]] = sramIntfType.readPorts.map(new FirrtlMemoryReader(_))
//     val firrtlWritePorts: Seq[FirrtlMemoryWriter[_]] = sramIntfType.writePorts.map(new FirrtlMemoryWriter(_))
//     val firrtlReadwritePorts: Seq[FirrtlMemoryReadwriter[_]] =
//       sramIntfType.readwritePorts.map(new FirrtlMemoryReadwriter(_))

//     // set references to firrtl memory ports
//     def nameAndSetRef(ports: Seq[Data], namePrefix: String): Seq[String] = {
//       ports.zipWithIndex.map { case (p, idx) =>
//         val name = namePrefix + idx
//         p.setRef(Slot(Node(mem), name))
//         name
//       }
//     }

//     val firrtlReadPortNames = nameAndSetRef(firrtlReadPorts, "R")
//     val firrtlWritePortNames = nameAndSetRef(firrtlWritePorts, "W")
//     val firrtlReadwritePortNames = nameAndSetRef(firrtlReadwritePorts, "RW")

//     // set bindings of firrtl memory ports
//     firrtlReadPorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))
//     firrtlWritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))
//     firrtlReadwritePorts.foreach(_.bind(SramPortBinding(Builder.forcedUserModule, Builder.currentBlock)))

//     // bind type so that memory type can get converted to FIRRTL
//     val boundType = tpe.cloneTypeFull
//     boundType.bind(FirrtlMemTypeBinding(mem))

//     // create FIRRTL memory
//     Builder.pushCommand(
//       FirrtlMemory(
//         sourceInfo,
//         mem,
//         boundType,
//         size,
//         firrtlReadPortNames,
//         firrtlWritePortNames,
//         firrtlReadwritePortNames
//       )
//     )

//     // connect firrtl memory ports to user-facing interface
//     for (((memReadPort, firrtlReadPort), readClock) <- _out.readPorts.zip(firrtlReadPorts).zip(readPortClocks)) {
//       firrtlReadPort.addr := memReadPort.address
//       firrtlReadPort.clk := readClock
//       memReadPort.data := firrtlReadPort.data.asInstanceOf[Data]
//       firrtlReadPort.en := memReadPort.enable
//     }
//     for (((memWritePort, firrtlWritePort), writeClock) <- _out.writePorts.zip(firrtlWritePorts).zip(writePortClocks)) {
//       firrtlWritePort.addr := memWritePort.address
//       firrtlWritePort.clk := writeClock
//       firrtlWritePort.data.asInstanceOf[Data] := memWritePort.data
//       firrtlWritePort.en := memWritePort.enable
//       assignMask(firrtlWritePort.mask, memWritePort.mask)
//     }
//     for (
//       ((memReadwritePort, firrtlReadwritePort), readwriteClock) <-
//         _out.readwritePorts.zip(firrtlReadwritePorts).zip(readwritePortClocks)
//     ) {
//       firrtlReadwritePort.addr := memReadwritePort.address
//       firrtlReadwritePort.clk := readwriteClock
//       firrtlReadwritePort.en := memReadwritePort.enable
//       memReadwritePort.readData := firrtlReadwritePort.rdata.asInstanceOf[Data]
//       firrtlReadwritePort.wdata.asInstanceOf[Data] := memReadwritePort.writeData
//       firrtlReadwritePort.wmode := memReadwritePort.isWrite
//       assignMask(firrtlReadwritePort.wmask, memReadwritePort.mask)
//     }
//     ModulePrefixAnnotation.annotate(mem)
//     _out
//   }

//   // There appears to be a bug in ScalaDoc where you cannot use macro-generated methods in the same
//   // compilation unit as the macro-generated type. This means that any use of Definition[_] or
//   // Instance[_] of the above classes within this compilation unit breaks ScalaDoc generation. This
//   // issue appears to be similar to https://stackoverflow.com/questions/42684101 but applying the
//   // specific mitigation did not seem to work.  As a workaround, we simply write the extension methods
//   // that are generated by the @instantiable macro so that we can use them here.
//   implicit class SRAMDescriptionInstanceMethods(underlying: Instance[SRAMDescription]) {
//     implicit val mg:       internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
//     def depthIn:           Property[BigInt] = underlying._lookup(_.depthIn)
//     def widthIn:           Property[Int] = underlying._lookup(_.widthIn)
//     def maskedIn:          Property[Boolean] = underlying._lookup(_.maskedIn)
//     def readIn:            Property[Int] = underlying._lookup(_.readIn)
//     def writeIn:           Property[Int] = underlying._lookup(_.writeIn)
//     def readwriteIn:       Property[Int] = underlying._lookup(_.readwriteIn)
//     def maskGranularityIn: Property[Int] = underlying._lookup(_.maskGranularityIn)
//     def hierarchyIn:       Property[Path] = underlying._lookup(_.hierarchyIn)
//   }

//   /** Assigns a given SRAM-style mask to a FIRRTL memory port mask. The FIRRTL
//     * memory port mask is tied to 1 for unmasked SRAMs.
//     *
//     * @param memWriteDataTpe write data type to get masked
//     * @param writeMaskOpt write mask to assign
//     */
//   private def assignMask(
//     writeMask:  SramMask,
//     maskSource: Option[Vec[Bool]]
//   ): Unit = {
//     maskSource match {
//       case None =>
//         // Write all 1s, all leaves are Bools
//         for (mask <- DataMirror.collectMembers(writeMask) { case b: Bool => b }) {
//           mask := true.B
//         }
//       case Some(source) =>
//         for ((maskElt, value) <- writeMask.vecElements.zip(source)) {
//           // All leaves are Bools, write the value from the maskOpt
//           for (leaf <- DataMirror.collectMembers(maskElt) { case b: Bool => b }) {
//             leaf := value
//           }
//         }
//     }
//   }

//   // Helper util to generate portedness descriptors based on the input parameters
//   // supplied to SRAM.apply
//   private[chisel3] def portedness(rd: Int, wr: Int, rw: Int): String = {
//     val rdPorts: String = if (rd > 0) s"${rd}R" else ""
//     val wrPorts: String = if (wr > 0) s"${wr}W" else ""
//     val rwPorts: String = if (rw > 0) s"${rw}RW" else ""

//     s"$rdPorts$wrPorts$rwPorts"
//   }
// }

/** Type representing the mask of an SRAM
  *
  * This type shadows the data type. It matches the structure of Aggregates with all leaves replaced with Bools.
  */
// private[chisel3] class SramMask(gen: Data) extends Record with OpaqueType {

//   override def opaqueType = gen match {
//     case r: Record => r._isOpaqueType
//     case _ => true
//   }

//   val elements = gen match {
//     case e: Element => ListMap("" -> Bool())
//     case v: Vec[_]  => ListMap("" -> Vec(v.length, new SramMask(v.sample_element)))
//     case r: Record  => r.elements.map { case (name, tpe) => name -> new SramMask(tpe) }
//   }

//   /** Used to assert that the SramMask and process its elements */
//   def vecElements: Vec[SramMask] = gen match {
//     case v: Vec[_] => elements.head._2.asInstanceOf[Vec[SramMask]]
//     case _ => Builder.exception(s"Internal Error! SramMask.vecElements called on non-Vec type $gen!")
//   }
// }

// /** Contains fields that map from the user-facing [[MemoryReadPort]] to a
//   * FIRRTL memory read port.
//   *
//   * @param readPort used to parameterize this class
//   *
//   * @note This is private because users should not directly use this bundle to
//   * interact with [[SRAM]].
//   */
// private[chisel3] final class FirrtlMemoryReader[T <: Data](readPort: MemoryReadPort[T]) extends Bundle {
//   val addr = readPort.address.cloneType
//   val en = Bool()
//   val clk = Clock()
//   val data = Flipped(readPort.data.cloneType)
// }

// /** Contains fields that map from the user-facing [[MemoryWritePort]] to a
//   * FIRRTL memory write port, excluding the mask, which is calculated and
//   * assigned explicitly.
//   *
//   * @param writePort used to parameterize this class
//   *
//   * @note This is private because users should not directly use this bundle to
//   * interact with [[SRAM]].
//   */
// private[chisel3] final class FirrtlMemoryWriter[T <: Data](writePort: MemoryWritePort[T]) extends Bundle {
//   val addr = writePort.address.cloneType
//   val en = Bool()
//   val clk = Clock()
//   val data = writePort.data.cloneType
//   val mask = new SramMask(data)
// }

// /** Contains fields that map from the user-facing [[MemoryReadwritePort]] to a
//   * FIRRTL memory read/write port, excluding the mask, which is calculated and
//   * assigned explicitly.
//   *
//   * @param readwritePort used to parameterize this class
//   *
//   * @note This is private because users should not directly use this bundle to
//   * interact with [[SRAM]].
//   */
// private[chisel3] final class FirrtlMemoryReadwriter[T <: Data](readwritePort: MemoryReadWritePort[T]) extends Bundle {
//   val addr = readwritePort.address.cloneType
//   val en = Bool()
//   val clk = Clock()
//   val rdata = Flipped(readwritePort.readData.cloneType)
//   val wmode = Bool()
//   val wdata = readwritePort.writeData.cloneType
//   val wmask = new SramMask(wdata)
// }