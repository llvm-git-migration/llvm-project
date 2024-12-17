// RUN: llvm-tblgen -gen-global-isel -I %p/../../include -I %p/Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

let Defs = [R0, R1] in
def tst1 : I<(outs), (ins), [(set R0, (get_fpenv))]>;

let Defs = [R0, R1] in
def tst2 : I<(outs GPR32:$rd), (ins GPR32:$rs1, GPR32:$rs2),
             [(set GPR32:$rd, R0, (udivrem i32:$rs1, i32:$rs2))]>;

def : Pat<(sdiv i32:$rs1, i32:$rs2), (tst2 $rs1, $rs2)>;
def : Pat<(sdivrem i32:$rs1, i32:$rs2), (tst2 $rs1, $rs2)>;

// CHECK-LABEL: // (sdiv:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst2:{ *:[i32] }:{ *:[i32] } ?:{ *:[i32] }:$rs1, ?:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::tst2),
// CHECK-NEXT: GIR_AddImplicitDef, /*InsnID*/0, GIMT_Encode2(MyTarget::R0), GIMT_Encode2(RegState::Dead),
// CHECK-NEXT: GIR_AddImplicitDef, /*InsnID*/0, GIMT_Encode2(MyTarget::R1), GIMT_Encode2(RegState::Dead),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 2,

// CHECK-LABEL: // (sdivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst2:{ *:[i32] }:{ *:[i32] } ?:{ *:[i32] }:$rs1, ?:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/2, // rs1
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/3, // rs2
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/0, /*OpIdx for MyTarget::R1*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 3,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // (udivrem:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)  =>  (tst2:{ *:[i32] }:{ *:[i32] } i32:{ *:[i32] }:$rs1, i32:{ *:[i32] }:$rs2)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[rd]
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/2, // rs1
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/3, // rs2
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/0, /*OpIdx for MyTarget::R1*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 1,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// CHECK-LABEL: // (get_fpenv:{ *:[i32] })  =>  (tst1:{ *:[i32] })
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::tst1),
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/0, /*OpIdx for MyTarget::R1*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // DstI[R0]
// CHECK-NEXT: GIR_AddRegister, /*InsnID*/1, GIMT_Encode2(MyTarget::R0), /*AddRegisterRegFlags*/GIMT_Encode2(0),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
