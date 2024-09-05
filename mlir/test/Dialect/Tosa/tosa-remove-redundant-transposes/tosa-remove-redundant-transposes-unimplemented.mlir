// RUN: mlir-opt --verify-diagnostics --split-input-file --verify-each --tosa-remove-redundant-transposes %s | FileCheck %s

// COM: we cannot do anything to the transpose in this case.
// CHECK-LABEL: @test_non_const_perms
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_non_const_perms(%perms: tensor<2xi32>) -> tensor<?x?xi32> {
  %0 = "tosa.const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = tosa.transpose %0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// COM: due to tracking back to a non-nullifying transpose, we can't get rid of the transposes entirely.
// COM: later editions of the pass may wish to fold these into a single transpose.
// CHECK-LABEL: @test_transpose_tracks_to_non_nullifying_transpose__single_step
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_transpose_tracks_to_non_nullifying_transpose__single_step(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x4x3xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x4x3x2xi32>
  %clamp = tosa.clamp %0 {min_int = 0 : i64, max_int = 1 : i64, min_fp = 0.0 : f64, max_fp = 1.0 : f64} : (tensor<1x4x3x2xi32>) -> tensor<1x4x3x2xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %clamp, %perms1 : (tensor<1x4x3x2xi32>, tensor<4xi32>) -> tensor<1x2x4x3xi32>
  return %1 : tensor<1x2x4x3xi32>
}

// -----

// COM: we don't deal with this case. --tosa-input-shapes is required.
// CHECK-LABEL: @test_unknown_dim_input_nullifying_pair
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unknown_dim_input_nullifying_pair(%arg0: tensor<3x?xi32>) -> tensor<3x2xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @test_unknown_dim__replacement_does_not_match
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unknown_dim__replacement_does_not_match(%arg0: tensor<3x?xi32>) -> tensor<?x?xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x?xi32>, tensor<2xi32>) -> tensor<?x3xi32>
  %1 = tosa.transpose %0, %perms : (tensor<?x3xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// COM: this would be able to be converted if --tosa-infer-shapes was run beforehand
// CHECK-LABEL: @test_unranked_tensors_present
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unranked_tensors_present(%arg0: tensor<3x2xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<3x2xi32>, tensor<2xi32>) -> tensor<*xi32>
  %1 = tosa.transpose %0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_unranked_everything
// CHECK: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_unranked_everything(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  %1 = tosa.transpose %0, %perms : (tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %1 : tensor<*xi32>
}

// -----

// COM: this is an example of some dead code we generate despite no transform taking place.
// COM: it will be removed by --canonicalize.
// COM: it's generated because at the add, we track back on the first argument first.
// CHECK-LABEL: @test_static_diverges_to_one_nullifying_one_non_nullifying
// CHECK: tosa.const
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: tosa.clamp
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: tosa.add
// CHECK-NEXT: tosa.const
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: return
func.func @test_static_diverges_to_one_nullifying_one_non_nullifying(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<1x2x4x3xi32>) -> tensor<1x2x3x4xi32> {
  %perms0 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %perms1 = "tosa.const"() {value = dense<[0, 3, 2, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose0 = tosa.transpose %arg0, %perms0 : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %transpose1 = tosa.transpose %arg1, %perms1 : (tensor<1x2x4x3xi32>, tensor<4xi32>) -> tensor<1x3x4x2xi32>
  %clamp = tosa.clamp %transpose0 {min_int = 0 : i64, max_int = 1 : i64, min_fp = 0.0 : f64, max_fp = 1.0 : f64} : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %abs = tosa.abs %transpose1 : (tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %add = tosa.add %clamp, %abs : (tensor<1x3x4x2xi32>, tensor<1x3x4x2xi32>) -> tensor<1x3x4x2xi32>
  %perms2 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %result = tosa.transpose %add, %perms2 : (tensor<1x3x4x2xi32>, tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %result : tensor<1x2x3x4xi32>
}
