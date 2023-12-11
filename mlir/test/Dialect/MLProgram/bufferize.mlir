// RUN: mlir-opt %s --mlprogram-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @global 
ml_program.global private mutable @global(dense<0> : tensor<i64>) : tensor<i64>

// CHECK-LABEL: @global_load_store
func.func @global_load_store() -> i64 {
  // CHECK-DAG: %[[CST127:.+]] = arith.constant 127
  // CHECK-DAG: %[[GLOBAL_1:.+]] = memref.get_global @global
  // CHECK-DAG: %[[NEW_ALLOC:.+]] = memref.alloc
  // CHECK:     memref.copy %[[GLOBAL_1]], %[[NEW_ALLOC]]
  // CHECK:     %[[TENSOR:.+]] = bufferization.to_tensor %[[NEW_ALLOC]]
  // CHECK:     %[[EXTRACTED:.+]] = tensor.extract %[[TENSOR]][]
  // CHECK:     %[[NEW_VALUE:.+]] = arith.muli %[[EXTRACTED]], %[[CST127]]
  // CHECK:     %[[INSERTED:.+]] = tensor.insert %[[NEW_VALUE]] into %[[TENSOR]][]
  // CHECK:     %[[GLOBAL_2:.+]] = memref.get_global @global
  // CHECK:     %[[MEMREF:.+]] = bufferization.to_memref %[[INSERTED]]
  // CHECK:     memref.copy %[[MEMREF]], %[[GLOBAL_2]]
  // CHECK:     return %[[NEW_VALUE]]
  %c127_i64 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : tensor<i64>
  %extracted = tensor.extract %0[] : tensor<i64>
  %1 = arith.muli %extracted, %c127_i64 : i64
  %inserted = tensor.insert %1 into %0[] : tensor<i64>
  ml_program.global_store @global = %inserted : tensor<i64>
  return %1 : i64
}

// -----

// expected-error @below {{unsupported global op type}}
ml_program.global private mutable @global(0 : i64) : i64

func.func @global_scalar() -> i64 {
  %c127_i64 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : i64
  %1 = arith.muli %0, %c127_i64 : i64
  ml_program.global_store @global = %1 : i64
  return %1 : i64
}

// -----

// expected-error @below {{unsupported global op type}}
ml_program.global private mutable @global(dense<0> : memref<i64>) : memref<i64>

func.func @global_memref() -> i64 {
  %c127_i64 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : memref<i64>
  %extracted = memref.load %0[] : memref<i64>
  %1 = arith.muli %extracted, %c127_i64 : i64
  memref.store %1, %0[] : memref<i64>
  ml_program.global_store @global = %0 : memref<i64>
  return %1 : i64
}

// -----

// expected-error @below {{invalid tensor element type}}
ml_program.global private mutable @global(dense<0> : tensor<memref<i64>>) : tensor<memref<i64>>

func.func @global_tensor_of_memref() -> i64 {
  %c127_i64 = arith.constant 127 : i64
  return %c127_i64 : i64
}

// -----

// expected-error @below {{unimplemented: global op bufferization with dynamic shape}}
ml_program.global private mutable @global(dense<0> : tensor<1xi64>) : tensor<?xi64>

func.func @global_dynamic_shape() -> i64 {
  %c127_i64 = arith.constant 127 : i64
  %c0 = arith.constant 0 : index
  %0 = ml_program.global_load @global : tensor<?xi64>
  %extracted = tensor.extract %0[%c0] : tensor<?xi64>
  %1 = arith.muli %extracted, %c127_i64 : i64
  %inserted = tensor.insert %1 into %0[%c0] : tensor<?xi64>
  ml_program.global_store @global = %inserted : tensor<?xi64>
  return %1 : i64
}
