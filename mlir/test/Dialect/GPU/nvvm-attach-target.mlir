// RUN: mlir-opt %s --nvvm-attach-target="" | FileCheck %s
// RUN: mlir-opt %s --nvvm-attach-target="-opt=1" | FileCheck %s -check-prefix=CHECK-OPTIONS

module attributes {gpu.container_module} {
  // CHECK-LABEL:gpu.binary @kernel_module1
  // CHECK:gpu.module @kernel_module1 [#nvvm.target]
  // CHECK-OPTIONS:gpu.module @kernel_module1 [#nvvm.target<flags = {libNVVMOptions = ["-opt=1"]}>]
  gpu.module @kernel_module1 {
    llvm.func @kernel(%arg0: i32, %arg1: !llvm.ptr,
        %arg2: !llvm.ptr, %arg3: i64, %arg4: i64,
        %arg5: i64) attributes {gpu.kernel} {
      llvm.return
    }
  }
}