//===-- RISCVInstrInfoZcmop.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Compressed
// May-Be-Operations Extension (Zcmop).
//
//===----------------------------------------------------------------------===//

multiclass tmpEclass<bits<3> imm3> {
def NAME#commonE : InstructionEncoding {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;
  let Size = 2;
  bit mayLoad      = 0;
  bit mayStore     = 0;
  bit hasSideEffects = 0;
  let Inst{1-0} = 0b01;
  let Inst{6-2} = 0;
  let Inst{7} = 0b1; //differ
  let Inst{10-8} = imm3;
  let Inst{12-11} = 0;
  let Inst{15-13} = 0b011;
}

def NAME#specialE : InstructionEncoding {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;
  let Size = 2;
  bit mayLoad      = 0;
  bit mayStore     = 0;
  bit hasSideEffects = 0;
  let Inst{1-0} = 0b01;
  let Inst{6-2} = 0;
  let Inst{7} = 0b0; //differ
  let Inst{10-8} = imm3;
  let Inst{12-11} = 0;
  let Inst{15-13} = 0b011;
}
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class CMOPInst<bits<3> imm3, string opcodestr>
    : RVInst16CI<0b011, 0b01, (outs), (ins), opcodestr, ""> {
  let Inst{6-2} = 0;
  let Inst{7} = 1;
  let Inst{10-8} = imm3;
  let Inst{12-11} = 0;
  let EncodingInfos = EncodingByHwMode<[DefaultMode, EncodingTmp],
                                       [!cast<InstructionEncoding>(NAME#commonE),
                                        !cast<InstructionEncoding>(NAME#specialE)]>;
}

// CMOP1, CMOP5 is used by Zicfiss.
let Predicates = [HasStdExtZcmop, NoHasStdExtZicfiss] in {
  defm CMOP1 : tmpEclass<0>;
  defm CMOP5 : tmpEclass<2>;
  def CMOP1 : CMOPInst<0, "cmop.1">, Sched<[]>;
  def CMOP5 : CMOPInst<2, "cmop.5">, Sched<[]>;
}

foreach n = [3, 7, 9, 11, 13, 15] in {
  let Predicates = [HasStdExtZcmop] in
  defm CMOP # n : tmpEclass<!srl(n, 1)>;
  def CMOP # n : CMOPInst<!srl(n, 1), "cmop." # n>, Sched<[]>;
}
