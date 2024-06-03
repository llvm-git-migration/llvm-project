// RUN: mlir-opt -convert-to-spirv %s | FileCheck %s

// CHECK-LABEL: @ub
// CHECK: %[[UNDEF0:.*]] = spirv.Undef : i32
// CHECK: spirv.ReturnValue %0 : i32
func.func @ub() -> index {
  %0 = ub.poison : index
  return %0 : index
}
