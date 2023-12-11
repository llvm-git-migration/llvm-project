// RUN: mlir-opt %s -one-shot-bufferize -split-input-file | FileCheck %s

// CHECK-LABEL: memref.global "private" @global 
ml_program.global private mutable @global(dense<0> : tensor<i64>) : tensor<i64>

// CHECK-LABEL: @global_load_store
func.func @global_load_store() -> i64 {
  // CHECK-DAG: %[[CST127:.+]] = arith.constant 127
  // CHECK-DAG: %[[GLOBAL_1:.+]] = memref.get_global @global
  // CHECK:     %[[VALUE:.+]] = memref.load %[[GLOBAL_1]][]
  // CHECK:     %[[NEW_VALUE:.+]] = arith.muli %[[VALUE]], %[[CST127]]
  // CHECK:     memref.store %[[NEW_VALUE]], %[[GLOBAL_1]][]
  // CHECK:     %[[GLOBAL_2:.+]] = memref.get_global @global
  // CHECK:     memref.copy %[[GLOBAL_1]], %[[GLOBAL_2]]
  // CHECK:     return %[[NEW_VALUE]]
  %c127_i64 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : tensor<i64>
  %extracted = tensor.extract %0[] : tensor<i64>
  %1 = arith.muli %extracted, %c127_i64 : i64
  %inserted = tensor.insert %1 into %0[] : tensor<i64>
  ml_program.global_store @global = %inserted : tensor<i64>
  return %1 : i64
}
