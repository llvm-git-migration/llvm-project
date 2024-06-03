// RUN: mlir-opt %s -convert-to-spirv | FileCheck %s

// CHECK-LABEL: @basic
func.func @basic(%a: index, %b: index) {
  // CHECK: spirv.IAdd
  %0 = index.add %a, %b
  // CHECK: spirv.ISub
  %1 = index.sub %a, %b
  // CHECK: spirv.IMul
  %2 = index.mul %a, %b
  // CHECK: spirv.SDiv
  %3 = index.divs %a, %b
  // CHECK: spirv.UDiv
  %4 = index.divu %a, %b
  // CHECK: spirv.SRem
  %5 = index.rems %a, %b
  // CHECK: spirv.UMod
  %6 = index.remu %a, %b
  // CHECK: spirv.GL.SMax
  %7 = index.maxs %a, %b
  // CHECK: spirv.GL.UMax
  %8 = index.maxu %a, %b
  // CHECK: spirv.GL.SMin
  %9 = index.mins %a, %b
  // CHECK: spirv.GL.UMin
  %10 = index.minu %a, %b
  // CHECK: spirv.ShiftLeftLogical
  %11 = index.shl %a, %b
  // CHECK: spirv.ShiftRightArithmetic
  %12 = index.shrs %a, %b
  // CHECK: spirv.ShiftRightLogical
  %13 = index.shru %a, %b
  // CHECK: spirv.BitwiseAnd
  %14 = index.and %a, %b
  // CHECK: spirv.BitwiseOr
  %15 = index.or %a, %b
  // CHECK: spirv.BitwiseXor
  %16 = index.xor %a, %b
  return
}

// CHECK-LABEL: @cmp
func.func @cmp(%a : index, %b : index) {
  // CHECK: spirv.IEqual
  %0 = index.cmp eq(%a, %b)
  // CHECK: spirv.INotEqual
  %1 = index.cmp ne(%a, %b)
  // CHECK: spirv.SLessThan
  %2 = index.cmp slt(%a, %b)
  // CHECK: spirv.SLessThanEqual
  %3 = index.cmp sle(%a, %b)
  // CHECK: spirv.SGreaterThan
  %4 = index.cmp sgt(%a, %b)
  // CHECK: spirv.SGreaterThanEqual
  %5 = index.cmp sge(%a, %b)
  // CHECK: spirv.ULessThan
  %6 = index.cmp ult(%a, %b)
  // CHECK: spirv.ULessThanEqual
  %7 = index.cmp ule(%a, %b)
  // CHECK: spirv.UGreaterThan
  %8 = index.cmp ugt(%a, %b)
  // CHECK: spirv.UGreaterThanEqual
  %9 = index.cmp uge(%a, %b)
  return
}

// CHECK-LABEL: @ceildivs
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK:    %[[ZERO:.*]] = spirv.Constant 0 : i32
// CHECK:    %[[ONE:.*]] = spirv.Constant 1 : i32
// CHECK:    %[[NEG1:.*]] = spirv.Constant -1 : i32
// CHECK:    %[[GREATER0:.*]] = spirv.SGreaterThan %[[ARG1]], %[[ZERO]] : i32
// CHECK:    %[[SELECT0:.*]]= spirv.Select %[[GREATER0]], %[[NEG1]], %[[ONE]] : i1, i32
// CHECK:    %[[IADD0:.*]] = spirv.IAdd %[[ARG0]], %[[SELECT0]]: i32
// CHECK:    %[[SDIV:.*]] = spirv.SDiv %[[IADD0]], %[[ARG1]] : i32
// CHECK:    %[[IADD1:.*]] = spirv.IAdd %[[SDIV]], %[[ONE]]  : i32
// CHECK:    %[[ISUB0:.*]] = spirv.ISub %[[ZERO]], %[[ARG0]] : i32
// CHECK:    %[[SDIV:.*]] = spirv.SDiv %[[ISUB0]], %[[ARG1]] : i32
// CHECK:    %[[ISUB1:.*]] = spirv.ISub %[[ZERO]], %[[SDIV]] : i32
// CHECK:    %[[GREATER1:.*]] = spirv.SGreaterThan %[[ARG0]], %[[ZERO]] : i32
// CHECK:    %[[LOGICALEQUAL:.*]] = spirv.LogicalEqual %[[GREATER1]], %[[GREATER0]] : i1
// CHECK:    %[[INOTEQUAL:.*]] = spirv.INotEqual %[[ARG0]], %[[ZERO]] : i32
// CHECK:    %[[LOGICALAND:.*]] = spirv.LogicalAnd %[[LOGICALEQUAL]], %[[INOTEQUAL]] : i1
// CHECK:    %[[SELECT1:.*]] = spirv.Select %[[LOGICALAND]], %[[IADD1]], %[[ISUB1]] : i1, i32
// CHECK:    spirv.ReturnValue %[[SELECT1]] : i32
func.func @ceildivs(%n: index, %m: index) -> index {
  %result = index.ceildivs %n, %m
  return %result : index
}

// CHECK-LABEL: @ceildivu
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK:    %[[ZERO:.*]] = spirv.Constant 0 : i32
// CHECK:    %[[ONE:.*]] = spirv.Constant 1 : i32
// CHECK:    %[[ISUB:.*]] = spirv.ISub %[[ARG0]], %[[ONE]] : i32
// CHECK:    %[[UDIV:.*]] = spirv.UDiv %[[ISUB]], %[[ARG1]] : i32
// CHECK:    %[[IADD:.*]] = spirv.IAdd %[[UDIV]], %[[ONE]] : i32
// CHECK:    %[[IEQUAL:.*]] = spirv.IEqual %[[ARG0]], %[[ZERO]] : i32
// CHECK:    %[[SELECT:.*]] = spirv.Select %[[IEQUAL]], %[[ZERO]], %[[IADD]] : i1, i32
// CHECK:    spirv.ReturnValue %[[SELECT]] : i32
func.func @ceildivu(%n: index, %m: index) -> index {
  %result = index.ceildivu %n, %m
  return %result : index
}
