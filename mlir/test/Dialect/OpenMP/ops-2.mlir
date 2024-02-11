// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

// CHECK-LABEL: parallel_op_privatizers
func.func @parallel_op_privatizers(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr, @y.privatizer %arg1 -> %arg3 : !llvm.ptr)
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr, @y.privatizer %arg1 -> %arg3 : !llvm.ptr) {
    %0 = llvm.load %arg2 : !llvm.ptr -> i32
    %1 = llvm.load %arg3 : !llvm.ptr -> i32
    omp.terminator
  }
  return
}

// CHECK: omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// CHECK: omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
omp.private {type = firstprivate} @y.privatizer : !llvm.ptr alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
// CHECK: } copy {
} copy {
// CHECK: ^bb0(%arg0: {{.*}}, %arg1: {{.*}}):
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
}

// CHECK-LABEL: parallel_op_reduction_and_private
func.func @parallel_op_reduction_and_private(%priv_var: !llvm.ptr, %priv_var2: !llvm.ptr, %reduc_var: !llvm.ptr, %reduc_var2: !llvm.ptr) {
  // CHECK: omp.parallel
  // CHECK-SAME: reduction(
  // CHECK-SAME: @add_f32 %[[reduc_var:[0-9a-z]+]] -> %[[reduc_arg:[0-9a-z]+]] : !llvm.ptr,
  // CHECK-SAME: @add_f32 %[[reduc_var2:[0-9a-z]+]] -> %[[reduc_arg2:[0-9a-z]+]] : !llvm.ptr)
  //
  // CHECK-SAME: private(
  // CHECK-SAME: @x.privatizer %[[priv_var:[0-9a-z]+]] -> %[[priv_arg:[0-9a-z]+]] : !llvm.ptr,
  // CHECK-SAME: @y.privatizer %[[priv_var2:[0-9a-z]+]] -> %[[priv_arg2:[0-9a-z]+]] : !llvm.ptr)
  omp.parallel reduction(@add_f32 %reduc_var -> %reduc_arg : !llvm.ptr, @add_f32 %reduc_var2 -> %reduc_arg2 : !llvm.ptr)
               private(@x.privatizer %priv_var -> %priv_arg : !llvm.ptr, @y.privatizer %priv_var2 -> %priv_arg2 : !llvm.ptr) {
    // CHECK: llvm.load %[[priv_arg]]
    %0 = llvm.load %priv_arg : !llvm.ptr -> f32
    // CHECK: llvm.load %[[priv_arg2]]
    %1 = llvm.load %priv_arg2 : !llvm.ptr -> f32
    // CHECK: llvm.load %[[reduc_arg]]
    %2 = llvm.load %reduc_arg : !llvm.ptr -> f32
    // CHECK: llvm.load %[[reduc_arg2]]
    %3 = llvm.load %reduc_arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  return
}

omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = arith.constant 0.0 : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = arith.addf %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
