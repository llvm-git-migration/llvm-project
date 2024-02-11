// RUN: mlir-opt -verify-diagnostics %s | mlir-opt | FileCheck %s

// CHECK-LABEL: parallel_op_privatizers
func.func @parallel_op_privatizers(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  // CHECK: omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr, @y.privatizer %arg1 -> %arg3 : !llvm.ptr)
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr, @y.privatizer %arg1 -> %arg3 : !llvm.ptr) {
  ^bb0(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
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
