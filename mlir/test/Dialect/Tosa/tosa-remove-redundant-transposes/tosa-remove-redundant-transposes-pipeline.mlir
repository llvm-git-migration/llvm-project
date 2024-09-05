// RUN: mlir-opt --verify-diagnostics --split-input-file --verify-each --tosa-remove-redundant-transposes --canonicalize %s | FileCheck %s
// COM: the purpose of this file is to demonstrate real examples of transformations on TOSA lowerings with the full
// COM: power of --tosa-remove-redundant-transposes in conjunction with --canonicalize

// COM: same as @test_resnet18_common_case in --tosa-remove-redundant-transposes-static-shapes

// CHECK-LABEL: @test_resnet18
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32_1> : tensor<64xf32>}> : () -> tensor<64xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32> : tensor<64xf32>}> : () -> tensor<64xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_6:.*]] = tosa.add %arg1, %[[VAL_4]] : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_7:.*]] = tosa.pow %[[VAL_6]], %[[VAL_5]] : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_8:.*]] = tosa.reciprocal %[[VAL_7]] : (tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG: %[[VAL_9:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_10:.*]] = tosa.sub %arg2, %[[VAL_9]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_11:.*]] = tosa.reshape %[[VAL_8]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_12:.*]] = tosa.mul %[[VAL_10]], %[[VAL_11]] {shift = 0 : i8} : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_13:.*]] = tosa.reshape %[[VAL_3]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_14:.*]] = tosa.mul %[[VAL_12]], %[[VAL_13]] {shift = 0 : i8} : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_15:.*]] = tosa.reshape %[[VAL_2]] {new_shape = array<i64: 1, 1, 1, 64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
// CHECK-DAG: %[[VAL_16:.*]] = tosa.add %[[VAL_14]], %[[VAL_15]] : (tensor<1x112x112x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK-DAG: %[[VAL_17:.*]] = tosa.clamp %[[VAL_16]] {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
// CHECK: return %[[VAL_17]] : tensor<1x112x112x64xf32>

func.func @test_resnet18(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %74: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {
    %59 = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32_1> : tensor<64xf32>}> : () -> tensor<64xf32>
    %60 = "tosa.const"() <{value = dense_resource<torch_tensor_64_torch.float32> : tensor<64xf32>}> : () -> tensor<64xf32>
    %63 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %64 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %69 = "tosa.const"() <{value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> tensor<1xf32>
    %70 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %75 = tosa.transpose %74, %64 : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %76 = tosa.add %arg1, %69 : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %77 = tosa.pow %76, %70 : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
    %78 = tosa.reciprocal %77 : (tensor<64xf32>) -> tensor<64xf32>
    %79 = tosa.reshape %arg0 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %80 = tosa.sub %75, %79 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %81 = tosa.reshape %78 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %82 = tosa.mul %80, %81 {shift = 0 : i8} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %83 = tosa.reshape %60 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %84 = tosa.mul %82, %83 {shift = 0 : i8} : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %85 = tosa.reshape %59 {new_shape = array<i64: 1, 64, 1, 1>} : (tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %86 = tosa.add %84, %85 : (tensor<1x64x112x112xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x112x112xf32>
    %87 = tosa.clamp %86 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %88 = tosa.transpose %87, %63 : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
    return %88 : tensor<1x112x112x64xf32>
}
