// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @unsafe_fp_math_func() 
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @unsafe_fp_math_func() attributes {unsafe_fp_math = true}  {
  llvm.return
}
// CHECK: attributes #[[ATTRS]] = { "unsafe-fp-math"="true" }
