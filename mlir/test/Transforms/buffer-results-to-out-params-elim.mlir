// RUN: mlir-opt -allow-unregistered-dialect -p 'builtin.module(buffer-results-to-out-params{avoid-buffer-result-alloc-copy})'  %s | FileCheck %s

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: memref<8x64xf32>) {
// CHECK-NOT:        memref.alloc()
// CHECK:           "test.source"(%[[ARG]])  : (memref<8x64xf32>) -> ()
// CHECK:           return
// CHECK:         }
func.func @basic() -> (memref<8x64xf32>) {
  %b = memref.alloc() : memref<8x64xf32>
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  return %b : memref<8x64xf32>
}

// CHECK-LABEL:   func @basic_no_change(
// CHECK-SAME:                %[[ARG:.*]]: memref<f32>) {
// CHECK:           %[[RESULT:.*]] = "test.source"() : () -> memref<f32>
// CHECK:           memref.copy %[[RESULT]], %[[ARG]]  : memref<f32> to memref<f32>
// CHECK:           return
// CHECK:         }
func.func @basic_no_change() -> (memref<f32>) {
  %0 = "test.source"() : () -> (memref<f32>)
  return %0 : memref<f32>
}