//===- WinogradConv2D.cpp - Winograd Conv2D implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Winograd Conv2D algorithm. The implementation is based on the
// paper: Fast Algorithms for Convolutional Neural Networks
// (https://arxiv.org/abs/1509.09308)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace linalg {

namespace {

using TransformMapKeyTy = std::pair<int, int>;

// We use F(m, r) to define the size of minimal filtering algorithms.
// m is the output dimension and r is the filter dimension. We can get
// the input dimension, alpha, from the formula, alpha = m + r - 1.
//
// For example, when m = 2 and r = 3, we know its input size is 4.
// The Conv2D will operate on 4x4 input data with 3x3 filter and get
// 2x2 output result.
constexpr TransformMapKeyTy F_2_3{2, 3};
constexpr TransformMapKeyTy F_4_3{4, 3};
constexpr TransformMapKeyTy F_2_5{2, 5};

Value collapse2DData(RewriterBase &rewriter, Location loc, Value data) {
  auto type = cast<ShapedType>(data.getType());
  auto elementType = type.getElementType();
  auto shape = type.getShape();
  auto collapseType = RankedTensorType::get(
      {shape[0] * shape[1], shape[2], shape[3]}, elementType);
  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}, {3}};
  return rewriter.create<tensor::CollapseShapeOp>(loc, collapseType, data,
                                                  reassociation);
}

// This function generates linalg.batch_matmul to multiply input with filter.
// linalg.batch_matmul only supports 3-dimension data sets. We can treat H x W
// data as the 1-dimension data array. That is to convert [H, W, N, C] to
// [H x W, N, C]. In this way, we can convert 4-dimension input data to
// 3-dimension representation that is suitable for linalg.batch_matmul.
//
// Batched matmul will do the matrix multiply with the reduction on channel.
//
// We get
//
// %collapsed_input = tensor.collapse_shape %input
// %collapsed_filter = tensor.collapse_shape %filter
// %ret = linalg.batch_matmul %collapsed_input, %collapsed_filter
// %expanded_ret = tensor.expand_shape %ret
//
// After this function, we get return value with data layout (H, W, N, F)
//
Value matrixMultiply(RewriterBase &rewriter, Location loc,
                     Value transformedFilter, Value transformedInput) {
  auto collapseFilter = collapse2DData(rewriter, loc, transformedFilter);
  auto collapseInput = collapse2DData(rewriter, loc, transformedInput);

  // Batched matrix multiply
  auto filterType = cast<ShapedType>(transformedFilter.getType());
  auto filterShape = filterType.getShape();
  auto inputType = cast<ShapedType>(transformedInput.getType());
  auto inputElemType = inputType.getElementType();
  auto inputShape = inputType.getShape();

  auto matmulType = RankedTensorType::get(
      {inputShape[0] * inputShape[1], inputShape[2], filterShape[3]},
      inputElemType);
  Value init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                inputElemType);

  auto matmulOp = rewriter.create<linalg::BatchMatmulOp>(
      loc, matmulType, ValueRange({collapseInput, collapseFilter}),
      ValueRange{init});

  // Expand matmul result
  SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}, {3}};
  auto expandType = RankedTensorType::get(
      {inputShape[0], inputShape[1], inputShape[2], filterShape[3]},
      inputElemType);
  auto expandOutput = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandType, matmulOp.getResult(0), reassociation);
  return expandOutput;
}

FailureOr<Operation *> winogradConv2DHelper(RewriterBase &rewriter,
                                            linalg::Conv2DNhwcFhwcOp convOp,
                                            int64_t m, int64_t r) {
  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  auto outputType = cast<ShapedType>(output.getType());
  int64_t outputH = outputType.getShape()[1];
  int64_t outputW = outputType.getShape()[2];
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape(); // F, H, W, C
  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape(); // N, H, W, C
  int64_t inputN = inputShape[0];
  int64_t inputC = inputShape[3];

  // Only support F(m x m, r x r), F(m x 1, r x 1) or F(1 x m, 1 x r)
  if ((outputH != outputW) && (outputH != 1 && outputW != 1))
    return failure();
  if ((filterH != filterW) && (filterH != 1 && filterW != 1))
    return failure();

  if ((outputH == 1 && filterH != 1) || (outputH != 1 && filterH == 1))
    return failure();
  if ((outputW == 1 && filterW != 1) || (outputW != 1 && filterW == 1))
    return failure();

  // Map from (m, r) to G transform matrix.
  static const llvm::SmallVector<TransformMapKeyTy, 3> validConfigs = {
      F_2_3, F_4_3, F_2_5};

  TransformMapKeyTy key = {m, r};
  auto it = std::find(validConfigs.begin(), validConfigs.end(), key);
  // If we cannot find the constant transformation matrix, it means we do
  // not support this configuration yet.
  if (it == validConfigs.end())
    return failure();

  // All the criterias are satisfied. We can do Winograd Conv2D.
  Location loc = convOp.getLoc();

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = outputH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = outputW != 1;

  // Create operator for filter transform
  Type elementType = filterType.getElementType();
  int64_t alphaH = leftTransform ? m + r - 1 : 1;
  int64_t alphaW = rightTransform ? m + r - 1 : 1;
  int64_t retHeight = leftTransform ? (outputH / m) * alphaH : 1;
  int64_t retWidth = rightTransform ? (outputW / m) * alphaW : 1;
  auto retType = RankedTensorType::get({retHeight, retWidth, filterC, filterF},
                                       elementType);
  Value retValue =
      rewriter.create<tensor::EmptyOp>(loc, retType.getShape(), elementType);
  auto transformedFilter = rewriter.create<linalg::WinogradFilterTransformOp>(
      loc, retType, filter, retValue, outputH, outputW, m, r);

  // Create operator for input transform
  retType =
      RankedTensorType::get({retHeight, retWidth, inputN, inputC}, elementType);
  retValue =
      rewriter.create<tensor::EmptyOp>(loc, retType.getShape(), elementType);
  auto transformedInput = rewriter.create<linalg::WinogradInputTransformOp>(
      loc, retType, input, retValue, outputH, outputW, m, r);

  Value matmulRet =
      matrixMultiply(rewriter, loc, transformedFilter, transformedInput);

  // create operator for output transform
  auto transformedOutput = rewriter.create<linalg::WinogradOutputTransformOp>(
      loc, outputType, matmulRet, output, m, r);

  rewriter.replaceOp(convOp, transformedOutput);

  return transformedOutput.getOperation();
}

class WinogradConv2DNhwcFhwc final
    : public OpRewritePattern<linalg::Conv2DNhwcFhwcOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  WinogradConv2DNhwcFhwc(mlir::MLIRContext *context, int64_t m, int64_t r)
      : OpRewritePattern(context), m(m), r(r) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcFhwcOp convOp,
                                PatternRewriter &rewriter) const override {
    Value filter = convOp.getInputs()[1];
    auto filterType = cast<ShapedType>(filter.getType());
    auto filterShape = filterType.getShape(); // F, H, W, C
    int64_t filterH = filterShape[1];
    int64_t filterW = filterShape[2];
    Value output = convOp.getOutputs()[0];
    auto outputType = cast<ShapedType>(output.getType());
    auto outputShape = outputType.getShape(); // F, H, W, C
    int64_t outputH = outputShape[1];
    int64_t outputW = outputShape[2];

    if (filterH != r && filterH != 1 && filterW != r && filterW != 1)
      return failure();

    if (outputH < m && outputH != 1 && outputW < m && outputW != 1)
      return failure();

    if (failed(winogradConv2DHelper(rewriter, convOp, m, r)))
      return failure();

    return success();
  }

private:
  int64_t m;
  int64_t r;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
FailureOr<Operation *> winogradConv2D(RewriterBase &rewriter,
                                      linalg::Conv2DNhwcFhwcOp op, int64_t m,
                                      int64_t r) {
  return winogradConv2DHelper(rewriter, op, m, r);
}

void populateWinogradConv2DPatterns(RewritePatternSet &patterns, int64_t m,
                                    int64_t r) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<WinogradConv2DNhwcFhwc>(context, m, r);
}

} // end namespace linalg
} // end namespace mlir
