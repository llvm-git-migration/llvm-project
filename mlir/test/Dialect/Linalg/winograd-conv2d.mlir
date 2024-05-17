// RUN: mlir-opt %s -split-input-file -test-linalg-transform-patterns=test-winograd-conv2d | FileCheck %s

func.func @conv2d_4x4_3x3(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4x4x2xf32> {
  %0 = tensor.empty() : tensor<2x4x4x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x4x4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x4x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x6x5xf32>, tensor<2x3x3x5xf32>) outs(%1 : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
  return %2 : tensor<2x4x4x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_4x4_3x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x6x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x4x4x2xf32> {
// CHECK:       %[[S0:.*]] = tensor.empty() : tensor<2x4x4x2xf32>
// CHECK-NEXT:  %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x4x4x2xf32>) {
// CHECK-NEXT:  ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:    linalg.yield %[[IN]] : f32
// CHECK-NEXT:  } -> tensor<2x4x4x2xf32>
// CHECK-NEXT:  %[[S2:.*]] = tensor.empty() : tensor<6x6x5x2xf32>
// CHECK-NEXT:  %[[S3:.*]] = linalg.winograd_filter_transform output_height(4) output_width(4) m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<6x6x5x2xf32>) -> tensor<6x6x5x2xf32>
// CHECK-NEXT:  %[[S4:.*]] = tensor.empty() : tensor<6x6x2x5xf32>
// CHECK-NEXT:  %[[S5:.*]] = linalg.winograd_input_transform output_height(4) output_width(4) m(4) r(3) ins(%[[ARG0]] : tensor<2x6x6x5xf32>) outs(%[[S4]] : tensor<6x6x2x5xf32>) -> tensor<6x6x2x5xf32>
// CHECK-NEXT:  %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2], [3]] : tensor<6x6x5x2xf32> into tensor<36x5x2xf32>
// CHECK-NEXT:  %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<6x6x2x5xf32> into tensor<36x2x5xf32>
// CHECK-NEXT:  %[[S6:.*]] = tensor.empty() : tensor<36x2x2xf32>
// CHECK-NEXT:  %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<36x2x5xf32>, tensor<36x5x2xf32>) outs(%[[S6]] : tensor<36x2x2xf32>) -> tensor<36x2x2xf32>
// CHECK-NEXT:  %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1], [2], [3]] output_shape [6, 6, 2, 2] : tensor<36x2x2xf32> into tensor<6x6x2x2xf32>
// CHECK-NEXT:  %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<6x6x2x2xf32>) outs(%[[S1]] : tensor<2x4x4x2xf32>) -> tensor<2x4x4x2xf32>
// CHECK-NEXT:  return %[[S8]] : tensor<2x4x4x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_2x2_3x3(%arg0: tensor<2x4x4x5xf32>, %arg1: tensor<2x3x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x2x2x2xf32> {
  %0 = tensor.empty() : tensor<2x2x2x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x2x2x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x4x4x5xf32>, tensor<2x3x3x5xf32>) outs(%1 : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %2 : tensor<2x2x2x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_2x2_3x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x4x4x5xf32>, %[[ARG1:.*]]: tensor<2x3x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x2x2x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x2x2x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x2x2x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x2x2x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<0x0x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform output_height(2) output_width(2) m(4) r(3) ins(%[[ARG1]] : tensor<2x3x3x5xf32>) outs(%[[S2]] : tensor<0x0x5x2xf32>) -> tensor<0x0x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<0x0x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform output_height(2) output_width(2) m(4) r(3) ins(%[[ARG0]] : tensor<2x4x4x5xf32>) outs(%[[S4]] : tensor<0x0x2x5xf32>) -> tensor<0x0x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2], [3]] : tensor<0x0x5x2xf32> into tensor<0x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<0x0x2x5xf32> into tensor<0x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<0x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<0x2x5xf32>, tensor<0x5x2xf32>) outs(%[[S6]] : tensor<0x2x2xf32>) -> tensor<0x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1], [2], [3]] output_shape [0, 0, 2, 2] : tensor<0x2x2xf32> into tensor<0x0x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<0x0x2x2xf32>) outs(%[[S1]] : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x2x2x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_2x2_5x5(%arg0: tensor<2x6x6x5xf32>, %arg1: tensor<2x5x5x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x2x2x2xf32> {
  %0 = tensor.empty() : tensor<2x2x2x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x2x2x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x6x5xf32>, tensor<2x5x5x5xf32>) outs(%1 : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %2 : tensor<2x2x2x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_2x2_5x5
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x6x5xf32>, %[[ARG1:.*]]: tensor<2x5x5x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x2x2x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x2x2x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x2x2x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x2x2x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<0x0x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform output_height(2) output_width(2) m(4) r(3) ins(%[[ARG1]] : tensor<2x5x5x5xf32>) outs(%[[S2]] : tensor<0x0x5x2xf32>) -> tensor<0x0x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<0x0x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform output_height(2) output_width(2) m(4) r(3) ins(%[[ARG0]] : tensor<2x6x6x5xf32>) outs(%[[S4]] : tensor<0x0x2x5xf32>) -> tensor<0x0x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2], [3]] : tensor<0x0x5x2xf32> into tensor<0x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<0x0x2x5xf32> into tensor<0x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<0x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<0x2x5xf32>, tensor<0x5x2xf32>) outs(%[[S6]] : tensor<0x2x2xf32>) -> tensor<0x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1], [2], [3]] output_shape [0, 0, 2, 2] : tensor<0x2x2xf32> into tensor<0x0x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<0x0x2x2xf32>) outs(%[[S1]] : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x2x2x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_1x4_1x3(%arg0: tensor<2x1x6x5xf32>, %arg1: tensor<2x1x3x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x1x4x2xf32> {
  %0 = tensor.empty() : tensor<2x1x4x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x1x4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x1x4x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x1x6x5xf32>, tensor<2x1x3x5xf32>) outs(%1 : tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32>
  return %2 : tensor<2x1x4x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_1x4_1x3
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x1x6x5xf32>, %[[ARG1:.*]]: tensor<2x1x3x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x1x4x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x1x4x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x1x4x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x1x4x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<1x6x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform output_height(1) output_width(4) m(4) r(3) ins(%[[ARG1]] : tensor<2x1x3x5xf32>) outs(%[[S2]] : tensor<1x6x5x2xf32>) -> tensor<1x6x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<1x6x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform output_height(1) output_width(4) m(4) r(3) ins(%[[ARG0]] : tensor<2x1x6x5xf32>) outs(%[[S4]] : tensor<1x6x2x5xf32>) -> tensor<1x6x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2], [3]] : tensor<1x6x5x2xf32> into tensor<6x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<1x6x2x5xf32> into tensor<6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S6]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1], [2], [3]] output_shape [1, 6, 2, 2] : tensor<6x2x2xf32> into tensor<1x6x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<1x6x2x2xf32>) outs(%[[S1]] : tensor<2x1x4x2xf32>) -> tensor<2x1x4x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x1x4x2xf32>
// CHECK-NEXT: }

// -----

func.func @conv2d_4x1_3x1(%arg0: tensor<2x6x1x5xf32>, %arg1: tensor<2x3x1x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4x1x2xf32> {
  %0 = tensor.empty() : tensor<2x4x1x2xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1xf32>) outs(%0 : tensor<2x4x1x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x4x1x2xf32>
  %2 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x6x1x5xf32>, tensor<2x3x1x5xf32>) outs(%1 : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
  return %2 : tensor<2x4x1x2xf32>
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @conv2d_4x1_3x1
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<2x6x1x5xf32>, %[[ARG1:.*]]: tensor<2x3x1x5xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> tensor<2x4x1x2xf32> {
// CHECK:        %[[S0:.*]] = tensor.empty() : tensor<2x4x1x2xf32>
// CHECK-NEXT:   %[[S1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG2]] : tensor<1xf32>) outs(%[[S0]] : tensor<2x4x1x2xf32>) {
// CHECK-NEXT:   ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK-NEXT:     linalg.yield %[[IN]] : f32
// CHECK-NEXT:   } -> tensor<2x4x1x2xf32>
// CHECK-NEXT:   %[[S2:.*]] = tensor.empty() : tensor<6x1x5x2xf32>
// CHECK-NEXT:   %[[S3:.*]] = linalg.winograd_filter_transform output_height(4) output_width(1) m(4) r(3) ins(%[[ARG1]] : tensor<2x3x1x5xf32>) outs(%[[S2]] : tensor<6x1x5x2xf32>) -> tensor<6x1x5x2xf32>
// CHECK-NEXT:   %[[S4:.*]] = tensor.empty() : tensor<6x1x2x5xf32>
// CHECK-NEXT:   %[[S5:.*]] = linalg.winograd_input_transform output_height(4) output_width(1) m(4) r(3) ins(%[[ARG0]] : tensor<2x6x1x5xf32>) outs(%[[S4]] : tensor<6x1x2x5xf32>) -> tensor<6x1x2x5xf32>
// CHECK-NEXT:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[S3]] {{\[}}[0, 1], [2], [3]] : tensor<6x1x5x2xf32> into tensor<6x5x2xf32>
// CHECK-NEXT:   %[[COLLAPSED_0:.*]] = tensor.collapse_shape %[[S5]] {{\[}}[0, 1], [2], [3]] : tensor<6x1x2x5xf32> into tensor<6x2x5xf32>
// CHECK-NEXT:   %[[S6:.*]] = tensor.empty() : tensor<6x2x2xf32>
// CHECK-NEXT:   %[[S7:.*]] = linalg.batch_matmul ins(%[[COLLAPSED_0]], %[[COLLAPSED]] : tensor<6x2x5xf32>, tensor<6x5x2xf32>) outs(%[[S6]] : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
// CHECK-NEXT:   %[[EXPANDED:.*]] = tensor.expand_shape %[[S7]] {{\[}}[0, 1], [2], [3]] output_shape [6, 1, 2, 2] : tensor<6x2x2xf32> into tensor<6x1x2x2xf32>
// CHECK-NEXT:   %[[S8:.*]] = linalg.winograd_output_transform m(4) r(3) ins(%[[EXPANDED]] : tensor<6x1x2x2xf32>) outs(%[[S1]] : tensor<2x4x1x2xf32>) -> tensor<2x4x1x2xf32>
// CHECK-NEXT:   return %[[S8]] : tensor<2x4x1x2xf32>
// CHECK-NEXT: }
