// RUN: mlir-opt --verify-diagnostics --split-input-file --verify-each --tosa-remove-redundant-transposes %s | FileCheck %s

// -----

// CHECK-LABEL: @test_unknown_dim_inner_replacement_matches
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose %arg0
// CHECK-NEXT: return %arg0
func.func @test_unknown_dim_inner_replacement_matches(%arg0: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<?x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<?x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}

// -----


// CHECK-LABEL: @test_unknown_dim_outer_replacement_matches
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose %arg0
// CHECK-NEXT: return %arg0
func.func @test_unknown_dim_outer_replacement_matches(%arg0: tensor<3x?xi32>) -> tensor<3x?xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x?xi32>
  return %1 : tensor<3x?xi32>
}

// -----

// CHECK-LABEL: @test_transpose_tracks_to_nullifying_diverging_binary_replacements_match
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose %arg0
// CHECK-NEXT: tosa.transpose %arg1
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: %[[CLAMP:.*]] = tosa.clamp %arg0
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: %[[ABS:.*]] = tosa.abs %arg1
// CHECK-NEXT: tosa.add
// CHECK-NEXT: %[[ADD:.*]] = tosa.add %[[CLAMP]], %[[ABS]]
// CHECK-NEXT: tosa.const
// CHECK-NOT: tosa.transpose
// CHECK-NEXT: return %[[ADD]]
func.func @test_transpose_tracks_to_nullifying_diverging_binary_replacements_match(%arg0: tensor<1x?x3x4xi32>, %arg1: tensor<1x2x?x4xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x?x3x4xi32>, tensor<4xi32>) -> tensor<?x3x4x?xi32>
  %transpose1 = tosa.transpose %arg1, %perms0 : (tensor<1x2x?x4xi32>, tensor<4xi32>) -> tensor<1x?x?x2xi32>
  %clamp = tosa.clamp %transpose0 {min_int = 0 : i64, max_int = 1 : i64, min_fp = 0.0 : f64, max_fp = 1.0 : f64} : (tensor<?x3x4x?xi32>) -> tensor<?x3x4x?xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x?x?x2xi32>) -> tensor<1x?x?x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<?x3x4x?xi32>, tensor<1x?x?x2xi32>) -> tensor<1x3x4x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms1 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}
