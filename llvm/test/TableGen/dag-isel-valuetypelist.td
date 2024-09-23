// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG : Register<"REG">;
def GPR : RegisterClass<"TestTarget", [i16, i32], 32, (add REG)>;

def int_foo : Intrinsic<[llvm_anyint_ty, llvm_anyint_ty], []>;

def INSTR_FOO_I16_I32 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}

def INSTR_FOO_I32_I16 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}

def i16_i32 : ValueTypeList<[i16, i32]>;
def i32_i16 : ValueTypeList<[i32, i16]>;

// CHECK:      OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_W_CHAIN)
// CHECK: 7*/  OPC_SwitchType {{.*}}, 10, /*MVT::i16*/6
// CHECK:       OPC_CheckTypeRes, 1, /*MVT::i32*/7
// CHECK:       OPC_MorphNodeTo2Chain, TARGET_VAL(::INSTR_FOO_I16_I32)
def : Pat<(i16_i32 (int_foo)), (i16_i32 (INSTR_FOO_I16_I32))>;

// CHECK: 20*/ /*SwitchType*/ {{.*}} /*MVT::i32*/7
// CHECK:       OPC_CheckTypeRes, 1, /*MVT::i16*/6
// CHECK:       OPC_MorphNodeTo2Chain, TARGET_VAL(::INSTR_FOO_I32_I16)
def : Pat<(i32_i16 (int_foo)), (i32_i16 (INSTR_FOO_I32_I16))>;
