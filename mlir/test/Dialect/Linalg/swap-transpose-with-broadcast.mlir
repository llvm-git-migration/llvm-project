//RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-swap-transpose-with-broadcast %s | FileCheck %s

func.func @broadcast_transpose_fold(%input: tensor<2x4x5xf32>,
                                    %init1: tensor<1x2x3x4x5x6xf32>,
                                    %init2: tensor<1x6x2x3x5x4xf32>) -> tensor<1x6x2x3x5x4xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<2x4x5xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<1x2x3x4x5x6xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<1x6x2x3x5x4xf32>
  //       CHECK:   %[[TMP_INIT:.+]] = tensor.empty() : tensor<2x5x4xf32>
  //       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<2x4x5xf32>) outs(%[[TMP_INIT]] : tensor<2x5x4xf32>) permutation = [0, 2, 1]
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[TRANSPOSE]] : tensor<2x5x4xf32>) outs(%[[INIT2]] : tensor<1x6x2x3x5x4xf32>) dimensions = [0, 3, 1]
  //       CHECK:   return %[[BROADCAST]] : tensor<1x6x2x3x5x4xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<2x4x5xf32>)
      outs(%init1 : tensor<1x2x3x4x5x6xf32>)
      dimensions = [0, 2, 5]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<1x2x3x4x5x6xf32>)
      outs(%init2 : tensor<1x6x2x3x5x4xf32>)
      permutation = [0, 5, 1, 2, 4, 3]
  func.return %transpose : tensor<1x6x2x3x5x4xf32>
}

// -----

func.func @broadcast_transpose_fold_dynamic(%input: tensor<?x?x5xf32>,
                                            %init1: tensor<1x?x3x?x5x6xf32>,
                                            %init2: tensor<1x3x?x6x5x?xf32>) -> tensor<1x3x?x6x5x?xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold_dynamic
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x5xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<1x?x3x?x5x6xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<1x3x?x6x5x?xf32>
  //   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
  //   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
  //       CHECK:   %[[DIM0:.+]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x5xf32>
  //       CHECK:   %[[DIM1:.+]] = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?x5xf32>
  //       CHECK:   %[[TMP_INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM0]]) : tensor<?x5x?xf32>
  //       CHECK:   %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[INPUT]] : tensor<?x?x5xf32>) outs(%[[TMP_INIT]] : tensor<?x5x?xf32>) permutation = [1, 2, 0]
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[TRANSPOSE]] : tensor<?x5x?xf32>) outs(%[[INIT2]] : tensor<1x3x?x6x5x?xf32>) dimensions = [0, 1, 3]
  //       CHECK:   return %[[BROADCAST]] : tensor<1x3x?x6x5x?xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<?x?x5xf32>)
      outs(%init1 : tensor<1x?x3x?x5x6xf32>)
      dimensions = [0, 2, 5]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<1x?x3x?x5x6xf32>)
      outs(%init2 : tensor<1x3x?x6x5x?xf32>)
      permutation = [0, 2, 3, 5, 4, 1]
  func.return %transpose : tensor<1x3x?x6x5x?xf32>
}

// -----

func.func @broadcast_transpose_fold_2dim(%input: tensor<2xf32>,
                                         %init1: tensor<2x4xf32>,
                                         %init2: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK-LABEL: @broadcast_transpose_fold_2dim
  //  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<2xf32>
  //  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<2x4xf32>
  //  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<4x2xf32>
  //       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[INPUT]] : tensor<2xf32>) outs(%[[INIT2]] : tensor<4x2xf32>) dimensions = [0]
  //       CHECK:   return %[[BROADCAST]] : tensor<4x2xf32>
  %broadcast = linalg.broadcast
      ins(%input : tensor<2xf32>)
      outs(%init1 : tensor<2x4xf32>)
      dimensions = [1]
  %transpose = linalg.transpose
      ins(%broadcast : tensor<2x4xf32>)
      outs(%init2 : tensor<4x2xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<4x2xf32>
}
