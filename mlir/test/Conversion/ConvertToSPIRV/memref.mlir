// RUN: mlir-opt -convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -cse -split-input-file %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_store_float_rank_zero
//  CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
//       CHECK: %[[CST0:.*]] = spirv.Constant 0 : i32
//       CHECK: %[[AC0:.*]] = spirv.AccessChain %[[ARG0]][%[[CST0]], %[[CST0]]] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK: %[[LOAD:.*]] = spirv.Load "StorageBuffer" %[[AC0]] : f32
//       CHECK: %[[AC1:.*]] = spirv.AccessChain %[[ARG1]][%[[CST0]], %[[CST0]]] : !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK: spirv.Store "StorageBuffer" %[[AC1]], %[[LOAD]] : f32
//       CHECK: spirv.Return
func.func @load_store_float_rank_zero(%arg0: memref<f32>, %arg1: memref<f32>) {
  %0 = memref.load %arg0[] : memref<f32>
  memref.store %0, %arg1[] : memref<f32>
  return
}

// CHECK-LABEL: @load_store_int_rank_one
//  CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer>, %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer>, %[[ARG2:.*]]: i32
//       CHECK: %[[CST0:.*]] = spirv.Constant 0 : i32
//       CHECK: %[[AC0:.*]] = spirv.AccessChain %[[ARG0]][%[[CST0]], %[[ARG2]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK: %[[LOAD:.*]] = spirv.Load "StorageBuffer" %[[AC0]] : i32
//       CHECK: %[[AC1:.*]] = spirv.AccessChain %[[ARG1]][%[[CST0]], %[[ARG2]]] : !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer>, i32, i32
//       CHECK: spirv.Store "StorageBuffer" %[[AC1]], %[[LOAD]] : i32
//       CHECK: spirv.Return
func.func @load_store_int_rank_one(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2 : index) {
  %0 = memref.load %arg0[%arg2] : memref<4xi32>
  memref.store %0, %arg1[%arg2] : memref<4xi32>
  return
}

} // end module