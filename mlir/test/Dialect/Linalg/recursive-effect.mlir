// RUN: mlir-opt %s --canonicalize | FileCheck %s

func.func @map(%arg0: memref<1xf32>, %arg1 : tensor<1xf32>) {
  %c1 = arith.constant 1 : index
  %init = arith.constant dense<0.0> : tensor<1xf32>
  %mapped = linalg.map ins(%arg1:tensor<1xf32>) outs(%init :tensor<1xf32>) 
            (%in : f32) {
              memref.store %in, %arg0[%c1] : memref<1xf32>
              linalg.yield %in : f32
            }
  func.return
}

// CHECK-LABEL: @map
//       CHECK: linalg.map
