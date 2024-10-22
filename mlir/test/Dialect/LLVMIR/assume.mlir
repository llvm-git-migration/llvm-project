// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @assume_align
// CHECK-SAME: (ptr %[[ARG:.+]])
llvm.func @assume_align(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i1) : i1
  %1 = llvm.mlir.constant(8 : i32) : i32
  // CHECK: call void @llvm.assume(i1 true) [ "align"(ptr %[[ARG]], i32 8) ]
  llvm.intr.assume.align %0, %arg0, %1 : (i1, !llvm.ptr, i32) -> ()
  llvm.return
}

// CHECK-LABEL: @assume_separate_storage
// CHECK-SAME: (ptr %[[ARG0:.+]], ptr %[[ARG1:.+]])
llvm.func @assume_separate_storage(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i1) : i1
  // CHECK: call void @llvm.assume(i1 true) [ "separate_storage"(ptr %[[ARG0]], ptr %[[ARG1]]) ]
  llvm.intr.assume.separate_storage %0, %arg0, %arg1 : (i1, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}
