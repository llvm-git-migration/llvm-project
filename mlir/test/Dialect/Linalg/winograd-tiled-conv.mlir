// RUN: mlir-opt %s -transform-interpreter -canonicalize | FileCheck %s

func.func @conv2d(%arg0: tensor<2x10x10x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
  %0 = tensor.empty() : tensor<2x8x8x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x8x8x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x8x8x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x10x10x5xf32>, tensor<2x3x3x5xf32>) outs(%1 : tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %2 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile_using_for %0 tile_sizes [0, 4, 4, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.winograd_conv2d %1 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @conv2d(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<2x10x10x5xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<2x3x3x5xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1xf32>) -> tensor<2x8x8x2xf32> {
// CHECK-NEXT:  %[[CST:.+]] = arith.constant 1.024000e+03 : f32
// CHECK-NEXT:  %[[CST_0:.+]] = arith.constant dense<{{\[}}[1.250000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01], [2.500000e-01, 2.500000e-01, 2.500000e-01, 2.500000e-01], [1.250000e-01, -2.500000e-01, 5.000000e-01, -1.000000e+00], [1.250000e-01, 2.500000e-01, 5.000000e-01, 1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 5.000000e-01]]> : tensor<6x4xf32>
// CHECK-NEXT:  %[[CST_1:.+]] = arith.constant dense<{{\[}}[1.250000e-01, 2.500000e-01, 2.500000e-01, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 2.500000e-01, 5.000000e-01, 5.000000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, 2.500000e-01, -1.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<4x6xf32>
// CHECK-NEXT:  %[[CST_2:.+]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, 2.500000e-01, -2.500000e-01, 2.500000e-01], [-3.125000e-01, -2.500000e-01, -2.500000e-01, -1.250000e-01, -1.250000e-01, 0.000000e+00], [0.000000e+00, -6.250000e-02, 6.250000e-02, -2.500000e-01, 2.500000e-01, -3.125000e-01], [6.250000e-02, 6.250000e-02, 6.250000e-02, 1.250000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-NEXT:  %[[CST_3:.+]] = arith.constant dense<{{\[}}[2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -2.500000e-01, -6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, -2.500000e-01, -2.500000e-01, 6.250000e-02, 6.250000e-02, 0.000000e+00], [0.000000e+00, 2.500000e-01, -1.250000e-01, -2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, -2.500000e-01, -1.250000e-01, 2.500000e-01, 1.250000e-01, 0.000000e+00], [0.000000e+00, 2.500000e-01, 0.000000e+00, -3.125000e-01, 0.000000e+00, 6.250000e-02]]> : tensor<6x6xf32>
// CHECK-NEXT:  %[[CST_4:.+]] = arith.constant dense<{{\[}}[1.000000e+00, -0.333333343, -0.333333343, 0.0833333358, 0.0833333358, 0.000000e+00], [0.000000e+00, 0.333333343, -0.333333343, -0.166666672, 0.166666672, 0.000000e+00], [0.000000e+00, -0.333333343, -0.333333343, 0.333333343, 0.333333343, 1.000000e+00]]> : tensor<3x6xf32>
// CHECK-NEXT:  %[[CST_5:.+]] = arith.constant dense<{{\[}}[1.000000e+00, 0.000000e+00, 0.000000e+00], [-0.333333343, 0.333333343, -0.333333343], [-0.333333343, -0.333333343, -0.333333343], [0.0833333358, -0.166666672, 0.333333343], [0.0833333358, 0.166666672, 0.333333343], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x3xf32>
// CHECK-NEXT:  %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:  %[[C5:.+]] = arith.constant 5 : index
// CHECK-NEXT:  %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:  %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT:  %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:  %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:  %[[S0:.+]] = tensor.empty() : tensor<2x8x8x2xf32>
// CHECK-NEXT:  %[[S1:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x8x8x2xf32>) {
// CHECK-NEXT:  ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:    linalg.yield %[[IN]] : f32
// CHECK-NEXT:  } -> tensor<2x8x8x2xf32>
// CHECK-NEXT:  %[[S2:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C8]] step %[[C4]] iter_args(%[[ARG4:.+]] = %[[S1]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:    %[[S3:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C8]] step %[[C4]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-NEXT:      %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[ARG3]], %[[ARG5]], 0] [2, 6, 6, 5] [1, 1, 1, 1] : tensor<2x10x10x5xf32> to tensor<2x6x6x5xf32>
// CHECK-NEXT:      %[[EXTRACTED_SLICE_6:.+]] = tensor.extract_slice %[[ARG6]][0, %[[ARG3]], %[[ARG5]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x8x8x2xf32> to tensor<2x4x4x2xf32>
// CHECK-NEXT:      %[[S4:.+]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK-NEXT:      %[[S5:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[S4]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:        %[[S11:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<6x6x5x2xf32>) {
// CHECK-NEXT:          %[[EXTRACTED_SLICE_8:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<2x3x3x5xf32> to tensor<1x3x3x1xf32>
// CHECK-NEXT:          %[[EXTRACTED_SLICE_9:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0] [1, 3, 3, 1] [1, 1, 1, 1] : tensor<1x3x3x1xf32> to tensor<3x3xf32>
// CHECK-NEXT:          %[[S12:.+]] = tensor.empty() : tensor<6x3xf32>
// CHECK-NEXT:          %[[S13:.+]] = linalg.matmul ins(%[[CST_5]], %[[EXTRACTED_SLICE_9]] : tensor<6x3xf32>, tensor<3x3xf32>) outs(%[[S12]] : tensor<6x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:          %[[S14:.+]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:          %[[S15:.+]] = linalg.matmul ins(%[[S13]], %[[CST_4]] : tensor<6x3xf32>, tensor<3x6xf32>) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:          %[[S16:.+]] = tensor.empty() : tensor<6x6x1x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_10:.+]] = tensor.insert_slice %[[S15]] into %[[S16]][0, 0, 0, 0] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x1x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_11:.+]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][0, 0, %[[ARG9]], %[[ARG7]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> into tensor<6x6x5x2xf32>
// CHECK-NEXT:          scf.yield %[[INSERTED_SLICE_11]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[S11]] : tensor<6x6x5x2xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[S6:.+]] = tensor.empty() : tensor<6x6x2x5xf32>
// CHECK-NEXT:      %[[S7:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[S6]]) -> (tensor<6x6x2x5xf32>) {
// CHECK-NEXT:        %[[S11:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<6x6x2x5xf32>) {
// CHECK-NEXT:          %[[EXTRACTED_SLICE_8:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<2x6x6x5xf32> to tensor<1x6x6x1xf32>
// CHECK-NEXT:          %[[EXTRACTED_SLICE_9:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x6x6x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:          %[[S12:.+]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:          %[[S13:.+]] = linalg.matmul ins(%[[CST_3]], %[[EXTRACTED_SLICE_9]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S12]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:          %[[S14:.+]] = tensor.empty() : tensor<6x6xf32>
// CHECK-NEXT:          %[[S15:.+]] = linalg.matmul ins(%[[S13]], %[[CST_2]] : tensor<6x6xf32>, tensor<6x6xf32>) outs(%[[S14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK-NEXT:          %[[S16:.+]] = tensor.empty() : tensor<6x6x1x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_10:.+]] = tensor.insert_slice %[[S15]] into %[[S16]][0, 0, 0, 0] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6xf32> into tensor<6x6x1x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_11:.+]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][0, 0, %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> into tensor<6x6x2x5xf32>
// CHECK-NEXT:          scf.yield %[[INSERTED_SLICE_11]] : tensor<6x6x2x5xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[S11]] : tensor<6x6x2x5xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:      %[[COLLAPSED_7:.+]] = tensor.collapse_shape %[[S7]] {{\[}}[0, 1], [2], [3]] : tensor<6x6x2x5xf32> into tensor<36x2x5xf32>
// CHECK-NEXT:      %[[S8:.+]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:      %[[S9:.+]] = linalg.batch_matmul ins(%[[COLLAPSED_7]], %[[COLLAPSED]] : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%[[S8]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:      %[[EXPANDED:.+]] = tensor.expand_shape %[[S9]] {{\[}}[0, 1], [2], [3]] output_shape [6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<6x6x2x2xf32>
// CHECK-NEXT:      %[[S10:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[EXTRACTED_SLICE_6]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:        %[[S11:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<2x4x4x2xf32>) {
// CHECK-NEXT:          %[[EXTRACTED_SLICE_8:.+]] = tensor.extract_slice %[[EXPANDED]][0, 0, %[[ARG7]], %[[ARG9]]] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x2x2xf32> to tensor<6x6x1x1xf32>
// CHECK-NEXT:          %[[EXTRACTED_SLICE_9:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE_8]][0, 0, 0, 0] [6, 6, 1, 1] [1, 1, 1, 1] : tensor<6x6x1x1xf32> to tensor<6x6xf32>
// CHECK-NEXT:          %[[S12:.+]] = tensor.empty() : tensor<4x6xf32>
// CHECK-NEXT:          %[[S13:.+]] = linalg.matmul ins(%[[CST_1]], %[[EXTRACTED_SLICE_9]] : tensor<4x6xf32>, tensor<6x6xf32>) outs(%[[S12]] : tensor<4x6xf32>) -> tensor<4x6xf32>
// CHECK-NEXT:          %[[S14:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:          %[[S15:.+]] = linalg.matmul ins(%[[S13]], %[[CST_0]] : tensor<4x6xf32>, tensor<6x4xf32>) outs(%[[S14]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:          %[[S16:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK-NEXT:          %[[S17:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]], #[[$MAP3]]], iterator_types = ["parallel", "parallel"]} ins(%[[CST]], %[[S15]] : f32, tensor<4x4xf32>) outs(%[[S16]] : tensor<4x4xf32>) {
// CHECK-NEXT:          ^bb0(%[[IN:.+]]: f32, %[[IN_12:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT:            %[[S19:.+]] = arith.mulf %[[IN]], %[[IN_12]] : f32
// CHECK-NEXT:            linalg.yield %[[S19]] : f32
// CHECK-NEXT:          } -> tensor<4x4xf32>
// CHECK-NEXT:          %[[S18:.+]] = tensor.empty() : tensor<1x4x4x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_10:.+]] = tensor.insert_slice %[[S17]] into %[[S18]][0, 0, 0, 0] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x4x4x1xf32>
// CHECK-NEXT:          %[[INSERTED_SLICE_11:.+]] = tensor.insert_slice %[[INSERTED_SLICE_10]] into %[[ARG10]][%[[ARG7]], 0, 0, %[[ARG9]]] [1, 4, 4, 1] [1, 1, 1, 1] : tensor<1x4x4x1xf32> into tensor<2x4x4x2xf32>
// CHECK-NEXT:          scf.yield %[[INSERTED_SLICE_11]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[S11]] : tensor<2x4x4x2xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[S10]] into %[[ARG6]][0, %[[ARG3]], %[[ARG5]], 0] [2, 4, 4, 2] [1, 1, 1, 1] : tensor<2x4x4x2xf32> into tensor<2x8x8x2xf32>
// CHECK-NEXT:      scf.yield %[[INSERTED_SLICE]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[S3]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[S2]] : tensor<2x8x8x2xf32>
// CHECK-NEXT:}
