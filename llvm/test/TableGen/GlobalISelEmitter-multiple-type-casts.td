// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def REG : Register<"REG">;
def GPR : RegisterClass<"MyTarget", [i16, i32], 32, (add REG)>;

def int_foo : Intrinsic<[llvm_anyint_ty, llvm_anyint_ty], []>;

def INSTR_FOO_I16_I32 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}

def INSTR_FOO_I32_I16 : Instruction {
  let OutOperandList = (outs GPR:$a, GPR:$b);
  let InOperandList = (ins);
}

// CHECK: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s16
// CHECK: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32
// CHECK: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(::INSTR_FOO_I16_I32)
def : Pat<([i16, i32] (int_foo)), ([i16, i32] (INSTR_FOO_I16_I32))>;

// CHECK: GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32
// CHECK: GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s16
// CHECK: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(::INSTR_FOO_I32_I16)
def : Pat<([i32, i16] (int_foo)), ([i32, i16] (INSTR_FOO_I32_I16))>;
