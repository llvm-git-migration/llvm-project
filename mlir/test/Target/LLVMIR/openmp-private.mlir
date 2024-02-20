// Test code-gen for `omp.parallel` ops with delayed privatizers (i.e. using
// `omp.private` ops).

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @parallel_op_1_private(%arg0: !llvm.ptr) {
  omp.parallel private(@x.privatizer %arg0 -> %arg2 : !llvm.ptr) {
//    %0 = llvm.load %arg2 : !llvm.ptr -> f32
    omp.terminator
  }
  llvm.return
}

omp.private {type = private} @x.privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr
  %1 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %1, %0 : f32, !llvm.ptr
  omp.yield(%0 : !llvm.ptr)
}
