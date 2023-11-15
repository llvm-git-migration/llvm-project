//==----- RISCVMacroFusion.td - Macro Fusion Definitions -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the macro fusion predicators.

class RISCVMacroFusion<list<Instruction> first,list<Instruction> second,
                             list<MCInstPredicate> extraFirstPreds = [],
                             list<MCInstPredicate> extraSecondPreds = []>
  : SimpleFusion<CheckAll<!listconcat([CheckOpcode<first>], extraFirstPreds)>,
                 CheckAll<!listconcat([CheckOpcode<second>], extraSecondPreds)>>;

def LUIADDI: RISCVMacroFusion<[LUI], [ADDI, ADDIW]>;

// clear upper 32 bits / get lower 32 bits: slli r1, r0, 32 + srli r1, r1, 32
def ClearUpper32Bits : RISCVMacroFusion<[SLLI], [SRLI],
                                        [CheckImmOperand<2, 32>],
                                        [CheckImmOperand<2, 32>]>;

// clear upper 48 bits / get lower 16 bits: slli r1, r0, 48 + srli r1, r1, 48
def ClearUpper48Bits : RISCVMacroFusion<[SLLI], [SRLI],
                                        [CheckImmOperand<2, 48>],
                                        [CheckImmOperand<2, 48>]>;

// clear upper 48 bits / get lower 16 bits: slliw r1, r0, 16 + srliw r1, r1, 16
def GetLower16Bits : RISCVMacroFusion<[SLLIW], [SRLIW],
                                      [CheckImmOperand<2, 16>],
                                      [CheckImmOperand<2, 16>]>;

// sign-extend a 16-bit number: slliw r1, r0, 16 + sraiw r1, r1, 16
def SExtH : RISCVMacroFusion<[SLLIW], [SRAIW],
                             [CheckImmOperand<2, 16>],
                             [CheckImmOperand<2, 16>]>;

// These should be covered by Zba extension?
// shift left by one and add: slli r1, r0, 1 + add r1, r1, r2
// shift left by two and add: slli r1, r0, 2 + add r1, r1, r2
// shift left by three and add: slli r1, r0, 3 + add r1, r1, r2
def ShiftNAdd : RISCVMacroFusion<[SLLI], [ADD],
                                 [CheckAny<[CheckImmOperand<2, 1>,
                                            CheckImmOperand<2, 2>,
                                            CheckImmOperand<2, 3>]>]>;

// shift zero-extended word left by one: slli r1, r0, 32 + srli r1, r0, 31
// shift zero-extended word left by two: slli r1, r0, 32 + srli r1, r0, 30
// shift zero-extended word left by three: slli r1, r0, 32 + srli r1, r0, 29
def ShiftZExtByN : RISCVMacroFusion<[SLLI], [SRLI],
                                    [CheckImmOperand<2, 32>],
                                    [CheckAny<[CheckImmOperand<2, 29>,
                                               CheckImmOperand<2, 30>,
                                               CheckImmOperand<2, 31>]>]>;

// get the second byte: srli r1, r0, 8 + andi r1, r1, 255
def GetSecondByte : RISCVMacroFusion<[SRLI], [ANDI],
                                     [CheckImmOperand<2, 8>],
                                     [CheckImmOperand<2, 255>]>;

// shift left by four and add: slli r1, r0, 4 + add r1, r1, r2
def ShiftLeft4Add : RISCVMacroFusion<[SLLI], [ADD], [CheckImmOperand<2, 4>]>;

// shift right by 29 and add: srli r1, r0, 29 + add r1, r1, r2
// shift right by 30 and add: srli r1, r0, 30 + add r1, r1, r2
// shift right by 31 and add: srli r1, r0, 31 + add r1, r1, r2
// shift right by 32 and add: srli r1, r0, 32 + add r1, r1, r2
def ShiftRightNAdd : RISCVMacroFusion<[SRLI], [ADD],
                                      [CheckAny<[CheckImmOperand<2, 29>,
                                                 CheckImmOperand<2, 30>,
                                                 CheckImmOperand<2, 31>,
                                                 CheckImmOperand<2, 32>]>]>;

// add one if odd, otherwise unchanged: andi r1, r0, 1 + add r1, r1, r2
// add one if odd (in word format), otherwise unchanged: andi r1, r0, 1 + addw r1, r1, r2
def AddOneIfOdd : RISCVMacroFusion<[ANDI], [ADD, ADDW], [CheckImmOperand<2, 1>]>;

// addw and extract its lower 8 bits (fused into addwbyte)
// addw and extract its lower 1 bit (fused into addwbit)
def AddAndExtractNBits : RISCVMacroFusion<[ADDW], [ANDI], [],
                                          [CheckAny<[CheckImmOperand<2, 1>,
                                                     CheckImmOperand<2, 255>]>]>;

// addw and zext.h (fused into addwzexth)
// addw and sext.h (fused into addwsexth)
def AddwAndExt : RISCVMacroFusion<[ADDW], [ZEXT_H_RV32, ZEXT_H_RV64, SEXT_H]>;

// logic operation and extract its LSB
def LogicOpAndExtractLSB : RISCVMacroFusion<[AND, OR, XOR, ANDI, ORI, XORI, ORC_B], [ANDI], [],  
                                            [CheckImmOperand<2, 1>]>;

// logic operation and extract its lower 16 bits
def LogicOpAndExtractLow16Bits : RISCVMacroFusion<[AND, OR, XOR, ANDI, ORI, XORI, ORC_B],
                                                  [ZEXT_H_RV32, ZEXT_H_RV64]>;

// OR(Cat(src1(63, 8), 0.U(8.W)), src2): andi r1, r0, -256 + or r1, r1, r2
def OrCat : RISCVMacroFusion<[ANDI], [OR], [CheckImmOperand<2, -256>]>;

// mul 7-bit data with 32-bit data: andi r1, r0, 127 + mulw r1, r1, r2
def Mul7BitsWith32Bits : RISCVMacroFusion<[ANDI], [MULW], [CheckImmOperand<2, 127>]>;
