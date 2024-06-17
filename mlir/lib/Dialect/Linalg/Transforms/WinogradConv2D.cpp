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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace linalg {

namespace {

// clang-format off
// Winograd Conv2D uses a minimal 2D filtering algorithm to calculate its
// result. The formula of minimal 2D filtering algorithm F(m x m, r x r),
// m is the output dimension and r is the filter dimension, is
//
// Y = A^T x [ (G x g x G^T) x (B^T x d x B) ] x A
//
// g is filter and d is input data. We need to prepare 6 constant
// transformation matrices, G, G^T, B^T, B, A^T, and A for this formula.
//
// The following tables define these constant transformation matrices for
// F(2 x 2, 3 x 3), F(4 x 4, 3 x 3), and F(2 x 2, 5 x 5)
constexpr float G_2x2_3x3[] = {
   -1,     0,   0,
 1./2, -1./2, 1./2,
 1./2,  1./2, 1./2,
    0,     0,    1
};

constexpr float GT_2x2_3x3[] = {
   -1,  1./2, 1./2, 0,
    0, -1./2, 1./2, 0,
    0,  1./2, 1./2, 1
};

constexpr float BT_2x2_3x3[] = {
   -1,    0,   1,   0,
    0,   -1,   1,   0,
    0,    1,   1,   0,
    0,   -1,   0,   1
};

constexpr float B_2x2_3x3[] = {
   -1,    0,   0,   0,
    0,   -1,   1,  -1,
    1,    1,   1,   0,
    0,    0,   0,   1
};

constexpr float AT_2x2_3x3[] = {
    1,    1,   1,   0,
    0,   -1,   1,   1
};

constexpr float A_2x2_3x3[] = {
    1,    0,
    1,   -1,
    1,    1,
    0,    1
};

constexpr float G_4x4_3x3[] = {
     1,     0,     0,
 -1./3,  1./3, -1./3,
 -1./3, -1./3, -1./3,
 1./12, -1./6,  1./3,
 1./12,  1./6,  1./3,
     0,     0,     1
};

constexpr float GT_4x4_3x3[] = {
 1,  -1./3, -1./3, 1./12, 1./12, 0,
 0,   1./3, -1./3, -1./6,  1./6, 0,
 0,  -1./3, -1./3,  1./3,  1./3, 1
};

constexpr float BT_4x4_3x3[] = {
 1./4,     0, -5./16,      0, 1./16,     0,
    0,  1./4,  -1./4, -1./16, 1./16,     0,
    0, -1./4,  -1./4,  1./16, 1./16,     0,
    0,  1./4,  -1./8,  -1./4,  1./8,     0,
    0, -1./4,  -1./8,   1./4,  1./8,     0,
    0,  1./4,      0, -5./16,     0, 1./16
};

constexpr float B_4x4_3x3[] = {
   1./4,      0,     0,     0,     0,      0,
      0,   1./4, -1./4,  1./4, -1./4,   1./4,
 -5./16,  -1./4, -1./4, -1./8, -1./8,      0,
      0, -1./16, 1./16, -1./4,  1./4, -5./16,
  1./16,  1./16, 1./16,  1./8,  1./8,      0,
      0,      0,     0,     0,     0,  1./16
};

constexpr float AT_4x4_3x3[] = {
 1./8,  1./4, 1./4,  1./8, 1./8,    0,
    0, -1./4, 1./4, -1./4, 1./4,    0,
    0,  1./4, 1./4,  1./2, 1./2,    0,
    0, -1./4, 1./4,    -1,    1, 1./2
};

constexpr float A_4x4_3x3[] = {
  1./8,     0,    0,     0,
  1./4, -1./4, 1./4, -1./4,
  1./4,  1./4, 1./4,  1./4,
  1./8, -1./4, 1./2,    -1,
  1./8,  1./4, 1./2,     1,
     0,     0,    0,  1./2
};

constexpr float G_2x2_5x5[] = {
     1,     0,      0,      0,      0,
  1./6, -1./6,   1./6,  -1./6,   1./6,
 -1./6, -1./6,  -1./6,  -1./6,  -1./6,
-4./15, 2./15, -1./15,  1./30, -1./60,
 1./60, 1./30,  1./15,  2./15,  4./15,
     0,     0,      0,      0,      1
};

constexpr float GT_2x2_5x5[] = {
   1,  1./6, -1./6, -4./15, 1./60, 0,
   0, -1./6, -1./6,  2./15, 1./30, 0,
   0,  1./6, -1./6, -1./15, 1./15, 0,
   0, -1./6, -1./6,  1./30, 2./15, 0,
   0,  1./6, -1./6, -1./60, 4./15, 1
};

constexpr float BT_2x2_5x5[] = {
 1./8,  3./16,  -1./4,  -3./16,   1./8,    0,
    0,   1./8,  1./16,  -5./16,   1./8,    0,
    0,  -1./8, -5./16,  -1./16,   1./8,    0,
    0,   1./4,  -1./8,   -1./4,   1./8,    0,
    0,  -1./8,  -1./4,    1./8,   1./4,    0,
    0,   1./8,  3./16,   -1./4, -3./16, 1./8
};

constexpr float B_2x2_5x5[] = {
   1./8,      0,      0,     0,     0,      0,
  3./16,   1./8,  -1./8,  1./4, -1./8,   1./8,
  -1./4,  1./16, -5./16, -1./8, -1./4,  3./16,
 -3./16, -5./16, -1./16, -1./4,  1./8,  -1./4,
   1./8,   1./8,   1./8,  1./8,  1./4, -3./16,
      0,      0,      0,     0,     0,   1./8
};

constexpr float AT_2x2_5x5[] = {
  1./2,  1, 1,  2, 1,    0,
     0, -1, 1, -1, 2, 1./2
};

constexpr float A_2x2_5x5[] = {
 1./2,    0,
    1,   -1,
    1,    1,
    2,   -1,
    1,    2,
    0, 1./2
};
// clang-format on

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

struct TransformMatrix {
  TransformMatrix(const float *table, int64_t rows, int64_t cols,
                  int64_t scalarFactor = 1)
      : table(table), rows(rows), cols(cols), scalarFactor(scalarFactor) {}

  const float *table;
  int64_t rows;
  int64_t cols;
  int64_t scalarFactor;
};

Value create2DTransformMatrix(RewriterBase &rewriter, Location loc,
                              TransformMatrix transform, Type type) {
  ArrayRef<float> const_vec(transform.table, transform.rows * transform.cols);

  return rewriter.create<arith::ConstantOp>(
      loc, DenseFPElementsAttr::get(
               RankedTensorType::get(
                   SmallVector<int64_t>{transform.rows, transform.cols}, type),
               const_vec));
}

Value extract2DData(RewriterBase &rewriter, Location loc, Value source,
                    Value outLoopIndex, Value inLoopIndex, int64_t outLoopIdx,
                    int64_t inLoopIdx, int64_t heightIdx, int64_t widthIdx) {
  auto sourceType = cast<ShapedType>(source.getType());
  Type elementType = sourceType.getElementType();
  auto sourceShape = sourceType.getShape();
  int64_t height = sourceShape[heightIdx];
  int64_t width = sourceShape[widthIdx];

  auto zeroIndex = rewriter.getIndexAttr(0);
  auto oneIndex = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult, 4> offsets(4, zeroIndex);
  offsets[outLoopIdx] = outLoopIndex;
  offsets[inLoopIdx] = inLoopIndex;
  SmallVector<OpFoldResult, 4> sizes(4, oneIndex);
  sizes[heightIdx] = rewriter.getIndexAttr(height);
  sizes[widthIdx] = rewriter.getIndexAttr(width);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);
  SmallVector<int64_t> targetShape(4, 1);
  targetShape[heightIdx] = height;
  targetShape[widthIdx] = width;

  auto targetType = RankedTensorType::get(targetShape, elementType);
  auto extractFilterOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, targetType, source, offsets, sizes, strides);

  auto extractFilterType = RankedTensorType::get({height, width}, elementType);
  auto extractFilter = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, extractFilterOp, extractFilterType);

  return extractFilter;
}

Value insert2DData(RewriterBase &rewriter, Location loc, Value source,
                   Value dest, Value outLoopIndex, Value inLoopIndex,
                   int64_t height, int64_t width, int64_t outLoopIdx,
                   int64_t inLoopIdx, int64_t heightIdx, int64_t widthIdx) {
  auto sourceType = cast<ShapedType>(source.getType());
  Type elementType = sourceType.getElementType();
  SmallVector<int64_t> sliceShape(4, 1);
  sliceShape[heightIdx] = height;
  sliceShape[widthIdx] = width;
  auto init = rewriter.create<tensor::EmptyOp>(loc, sliceShape, elementType);
  auto result = tensor::createCanonicalRankReducingInsertSliceOp(rewriter, loc,
                                                                 source, init);

  auto zeroIndex = rewriter.getIndexAttr(0);
  auto oneIndex = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult, 4> retOffsets(4, zeroIndex);
  retOffsets[outLoopIdx] = outLoopIndex;
  retOffsets[inLoopIdx] = inLoopIndex;
  SmallVector<OpFoldResult, 4> retSizes(4, oneIndex);
  retSizes[heightIdx] = rewriter.getIndexAttr(height);
  retSizes[widthIdx] = rewriter.getIndexAttr(width);
  SmallVector<OpFoldResult, 4> strides(4, oneIndex);

  auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      loc, result, dest, retOffsets, retSizes, strides);

  return insertSliceOp;
}

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

// This function transforms the filter. The data layout of the filter is FHWC.
// The transformation matrix is 2-dimension. We need to extract H x W from
// FHWC first. We need to generate 2 levels of loops to iterate on F and C.
// After the transformation, we get
//
// scf.for %f = lo_f to hi_f step 1
//   scf.for %c = lo_c to hi_c step 1
//     %extracted = extract filter<h x w> from filter<f x h x w x c>
//     %ret = linalg.matmul G, %extracted
//     %ret = linalg.matmul %ret, GT
//     %inserted = insert %ret into filter<h x w x c x f>
//
Value filterTransform(RewriterBase &rewriter, Location loc, Value filter,
                      Value retValue, int64_t m, int64_t r,
                      bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to G transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      GMatrices = {
          {F_2_3, TransformMatrix(G_2x2_3x3, 4, 3)},
          {F_4_3, TransformMatrix(G_4x4_3x3, 6, 3)},
          {F_2_5, TransformMatrix(G_2x2_5x5, 6, 5)},
      };

  // Map from (m, r) to GT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      GTMatrices = {
          {F_2_3, TransformMatrix(GT_2x2_3x3, 3, 4)},
          {F_4_3, TransformMatrix(GT_4x4_3x3, 3, 6)},
          {F_2_5, TransformMatrix(GT_2x2_5x5, 5, 6)},
      };

  auto filterType = cast<ShapedType>(filter.getType());
  Type elementType = filterType.getElementType();
  auto filterShape = filterType.getShape(); // F, H, W, C
  int64_t filterF = filterShape[0];
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];
  int64_t filterC = filterShape[3];

  if (filterH != r && filterH != 1)
    return Value();
  if (filterW != r && filterW != 1)
    return Value();

  // Return shape is <H x W x C x F>
  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto fUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, filterF);
  auto cUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, filterC);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto outerForOp =
      rewriter.create<scf::ForOp>(loc, zeroIdx, fUpperBound, oneStep, retValue);
  Block *outerForBody = outerForOp.getBody();
  rewriter.setInsertionPointToStart(outerForBody);
  Value FIter = outerForBody->getArgument(0);

  auto innerForOp = rewriter.create<scf::ForOp>(
      loc, zeroIdx, cUpperBound, oneStep, outerForOp.getRegionIterArgs()[0]);
  Block *innerForBody = innerForOp.getBody();
  rewriter.setInsertionPointToStart(innerForBody);
  Value CIter = innerForBody->getArgument(0);

  // Extract (H, W) from (F, H, W, C)
  auto extractFilter =
      extract2DData(rewriter, loc, filter, FIter, CIter, /*outLoopIdx=*/0,
                    /*inLoopIdx=*/3, /*heightIdx=*/1, /*widthIdx=*/2);

  TransformMapKeyTy key = {m, r};
  int64_t retRows = 1;
  Value matmulRetValue = extractFilter;
  if (leftTransform) {
    // Get constant transform matrix G
    auto it = GMatrices.find(key);
    if (it == GMatrices.end())
      return Value();
    const TransformMatrix &GMatrix = it->second;

    retRows = GMatrix.rows;
    auto matmulType = RankedTensorType::get({retRows, filterW}, elementType);
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);

    Value G = create2DTransformMatrix(rewriter, loc, GMatrix, elementType);
    // Multiply G x g
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{G, extractFilter}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  if (rightTransform) {
    // Get constant transform matrix GT
    auto it = GTMatrices.find(key);
    if (it == GTMatrices.end())
      return Value();
    const TransformMatrix &GTMatrix = it->second;

    auto matmulType =
        RankedTensorType::get({retRows, GTMatrix.cols}, elementType);
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);

    Value GT = create2DTransformMatrix(rewriter, loc, GTMatrix, elementType);
    // Multiply u = (G x g) x GT
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{matmulRetValue, GT}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  // Insert (H, W) to (H, W, C, F)
  Value iterArg = innerForOp.getRegionIterArgs()[0];
  int64_t retHeight = leftTransform ? m + r - 1 : 1;
  int64_t retWidth = rightTransform ? m + r - 1 : 1;
  auto insertSliceOp = insert2DData(
      rewriter, loc, matmulRetValue, iterArg, FIter, CIter, retHeight, retWidth,
      /*outLoopIdx=*/3, /*inLoopIdx=*/2, /*heightIdx=*/0, /*widthIdx=*/1);

  rewriter.create<scf::YieldOp>(loc, insertSliceOp);

  rewriter.setInsertionPointToEnd(outerForBody);
  rewriter.create<scf::YieldOp>(loc, innerForOp.getResult(0));

  rewriter.setInsertionPointAfter(outerForOp);

  return outerForOp.getResult(0);
}

// This function transforms the input. The data layout of the input is NHWC.
// The transformation matrix is 2-dimension. We need to extract H x W from
// NHWC first. We need to generate 2 levels of loops to iterate on N and C.
// After the transformation, we get
//
// scf.for %n = lo_n to hi_n step 1
//   scf.for %c = lo_c to hi_c step 1
//     %extracted = extract input<h x w> from input<n x h x w x c>
//     %ret = linalg.matmul BT, %extracted
//     %ret = linalg.matmul %ret, B
//     %inserted = insert %ret into input<h x w x n x c>
//
Value inputTransform(RewriterBase &rewriter, Location loc, Value input,
                     Value retValue, int64_t m, int64_t r,
                     bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to BT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      BTMatrices = {
          {F_2_3, TransformMatrix(BT_2x2_3x3, 4, 4)},
          {F_4_3, TransformMatrix(BT_4x4_3x3, 6, 6)},
          {F_2_5, TransformMatrix(BT_2x2_5x5, 6, 6)},
      };

  // Map from (m, r) to B transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      BMatrices = {
          {F_2_3, TransformMatrix(B_2x2_3x3, 4, 4)},
          {F_4_3, TransformMatrix(B_4x4_3x3, 6, 6)},
          {F_2_5, TransformMatrix(B_2x2_5x5, 6, 6)},
      };

  auto inputType = cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();
  auto inputShape = inputType.getShape(); // N, H, W, C
  int64_t inputN = inputShape[0];
  int64_t inputH = inputShape[1];
  int64_t inputW = inputShape[2];
  int64_t inputC = inputShape[3];
  int64_t alphaH = leftTransform ? m + r - 1 : 1;
  int64_t alphaW = rightTransform ? m + r - 1 : 1;

  if (inputH != alphaH && inputH != 1)
    return Value();
  if (inputW != alphaW && inputW != 1)
    return Value();

  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto nUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, inputN);
  auto cUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, inputC);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto outerForOp =
      rewriter.create<scf::ForOp>(loc, zeroIdx, nUpperBound, oneStep, retValue);
  Block *outerForBody = outerForOp.getBody();
  rewriter.setInsertionPointToStart(outerForBody);
  Value NIter = outerForBody->getArgument(0);

  auto innerForOp = rewriter.create<scf::ForOp>(
      loc, zeroIdx, cUpperBound, oneStep, outerForOp.getRegionIterArgs()[0]);
  Block *innerForBody = innerForOp.getBody();
  rewriter.setInsertionPointToStart(innerForBody);
  Value CIter = innerForBody->getArgument(0);

  // Extract (H, W) from (N, H, W, C)
  auto extractInput =
      extract2DData(rewriter, loc, input, NIter, CIter, /*outLoopIdx=*/0,
                    /*inLoopIdx=*/3, /*heightIdx=*/1, /*widthIdx=*/2);

  TransformMapKeyTy key = {m, r};
  int64_t retRows = 1;
  int64_t retCols = 1;
  Value matmulRetValue = extractInput;
  if (leftTransform) {
    // Get constant transform matrix BT
    auto it = BTMatrices.find(key);
    if (it == BTMatrices.end())
      return Value();
    const TransformMatrix &BTMatrix = it->second;

    retRows = BTMatrix.rows;
    auto matmulType = RankedTensorType::get({retRows, inputW}, elementType);
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);

    Value BT =
        create2DTransformMatrix(rewriter, loc, BTMatrix, rewriter.getF32Type());
    // Multiply BT x d
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{BT, matmulRetValue}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  if (rightTransform) {
    // Get constant transform matrix B
    auto it = BMatrices.find(key);
    if (it == BMatrices.end())
      return Value();
    const TransformMatrix &BMatrix = it->second;

    retCols = BMatrix.cols;
    auto matmulType = RankedTensorType::get({retRows, retCols}, elementType);
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);
    Value B =
        create2DTransformMatrix(rewriter, loc, BMatrix, rewriter.getF32Type());
    // Multiply v = (BT x d) x B
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{matmulRetValue, B}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  // Insert v
  // Insert (H, W) to (H, W, N, C)
  Value iterArg = innerForOp.getRegionIterArgs()[0];
  auto combinedVal = insert2DData(
      rewriter, loc, matmulRetValue, iterArg, NIter, CIter, retRows, retCols,
      /*outLoopIdx=*/2, /*inLoopIdx=*/3, /*heightIdx=*/0, /*widthIdx=*/1);

  rewriter.create<scf::YieldOp>(loc, combinedVal);

  rewriter.setInsertionPointToEnd(outerForBody);
  rewriter.create<scf::YieldOp>(loc, innerForOp.getResult(0));

  rewriter.setInsertionPointAfter(outerForOp);

  return outerForOp.getResult(0);
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

// This function transforms the output. The data layout of the output is HWNF.
// The transformation matrix is 2-dimension. We need to extract H x W from
// HWNF first. We need to generate 2 levels of loops to iterate on N and F.
// After the transformation, we get
//
// scf.for %n = lo_n to hi_n step 1
//   scf.for %f = lo_f to hi_f step 1
//     %extracted = extract input<h x w> from result<h x w x n x f>
//     %ret = linalg.matmul AT, %extracted
//     %ret = linalg.matmul %ret, A
//     %inserted = insert %ret into ret<n x h x w x f>
//
Value outputTransform(RewriterBase &rewriter, Location loc, Value value,
                      Value output, int64_t m, int64_t r,
                      bool leftTransform = true, bool rightTransform = true) {
  // Map from (m, r) to AT transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      ATMatrices = {
          {F_2_3, TransformMatrix(AT_2x2_3x3, 2, 4)},
          {F_4_3, TransformMatrix(AT_4x4_3x3, 4, 6, 32)},
          {F_2_5, TransformMatrix(AT_2x2_5x5, 2, 6, 16)},
      };

  // Map from (m, r) to A transform matrix.
  static const llvm::SmallDenseMap<TransformMapKeyTy, TransformMatrix>
      AMatrices = {
          {F_2_3, TransformMatrix(A_2x2_3x3, 4, 2)},
          {F_4_3, TransformMatrix(A_4x4_3x3, 6, 4, 32)},
          {F_2_5, TransformMatrix(A_2x2_5x5, 6, 2, 16)},
      };

  auto valueType = cast<ShapedType>(value.getType());
  Type elementType = valueType.getElementType();
  auto valueShape = valueType.getShape(); // H, W, N, F
  int64_t valueH = valueShape[0];
  int64_t valueW = valueShape[1];
  int64_t valueN = valueShape[2];
  int64_t valueF = valueShape[3];
  int64_t alphaH = leftTransform ? m + r - 1 : 1;
  int64_t alphaW = rightTransform ? m + r - 1 : 1;

  if (valueH != alphaH && valueH != 1)
    return Value();
  if (valueW != alphaW && valueW != 1)
    return Value();

  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto nUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, valueN);
  auto fUpperBound = rewriter.create<arith::ConstantIndexOp>(loc, valueF);
  auto oneStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  auto outerForOp =
      rewriter.create<scf::ForOp>(loc, zeroIdx, nUpperBound, oneStep, output);
  Block *outerForBody = outerForOp.getBody();
  rewriter.setInsertionPointToStart(outerForBody);
  Value NIter = outerForBody->getArgument(0);

  auto innerForOp = rewriter.create<scf::ForOp>(
      loc, zeroIdx, fUpperBound, oneStep, outerForOp.getRegionIterArgs()[0]);
  Block *innerForBody = innerForOp.getBody();
  rewriter.setInsertionPointToStart(innerForBody);
  Value FIter = innerForBody->getArgument(0);

  // Extract (H, W) from (H, W, N, F)
  auto extractValue =
      extract2DData(rewriter, loc, value, NIter, FIter, /*outLoopIdx=*/2,
                    /*inLoopIdx=*/3, /*heightIdx=*/0, /*widthIdx=*/1);

  TransformMapKeyTy key = {m, r};
  int64_t retRows = 1;
  int64_t retCols = 1;
  int64_t leftScalarFactor = 1;
  int64_t rightScalarFactor = 1;
  Value matmulRetValue = extractValue;
  if (leftTransform) {
    // Get constant transform matrix AT
    auto it = ATMatrices.find(key);
    if (it == ATMatrices.end())
      return Value();
    const TransformMatrix &ATMatrix = it->second;

    leftScalarFactor = ATMatrix.scalarFactor;
    retRows = ATMatrix.rows;
    auto matmulType = RankedTensorType::get({retRows, valueW}, elementType);
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);

    Value AT = create2DTransformMatrix(rewriter, loc, ATMatrix, elementType);
    // Multiply AT x m
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{AT, matmulRetValue}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  if (rightTransform) {
    // Get constant transform matrix T
    auto it = AMatrices.find(key);
    if (it == AMatrices.end())
      return Value();
    const TransformMatrix &AMatrix = it->second;

    rightScalarFactor = AMatrix.scalarFactor;
    auto matmulType =
        RankedTensorType::get({retRows, AMatrix.cols}, elementType);
    retCols = AMatrix.cols;
    auto init = rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(),
                                                 elementType);

    Value A = create2DTransformMatrix(rewriter, loc, AMatrix, elementType);
    // Multiply y = (AT x m) x A
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, matmulType, ValueRange{matmulRetValue, A}, ValueRange{init});
    matmulRetValue = matmulOp.getResult(0);
  }

  // Multiply scalar factor.
  Value scalarFactor = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(elementType, leftScalarFactor * rightScalarFactor));
  auto matmulType = RankedTensorType::get({retRows, retCols}, elementType);
  auto init =
      rewriter.create<tensor::EmptyOp>(loc, matmulType.getShape(), elementType);

  auto identityAffineMap = rewriter.getMultiDimIdentityMap(2);
  SmallVector<AffineMap> affineMaps = {AffineMap::get(2, 0, init.getContext()),
                                       identityAffineMap, identityAffineMap};
  auto scalarMatrixOp = rewriter.create<linalg::GenericOp>(
      loc, matmulType, ValueRange{scalarFactor, matmulRetValue},
      ValueRange{init}, affineMaps, tosa::getNParallelLoopsAttrs(2),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value scalarVal = args[0];
        Value matrixVal = args[1];
        Value result = nestedBuilder.create<arith::MulFOp>(nestedLoc, scalarVal,
                                                           matrixVal);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
      });

  // Insert slice y
  // Insert (H, W) to (N, H, W, F)
  Value iterArg = innerForOp.getRegionIterArgs()[0];
  Value combinedVal =
      insert2DData(rewriter, loc, scalarMatrixOp.getResult(0), iterArg, NIter,
                   FIter, retRows, retCols,
                   /*outLoopIdx=*/0,
                   /*inLoopIdx=*/3, /*heightIdx=*/1, /*widthIdx=*/2);

  rewriter.create<scf::YieldOp>(loc, combinedVal);

  rewriter.setInsertionPointToEnd(outerForBody);
  rewriter.create<scf::YieldOp>(loc, innerForOp.getResult(0));

  rewriter.setInsertionPointAfter(outerForOp);

  return outerForOp.getResult(0);
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

FailureOr<Operation *>
decomposeWinogradFilterTransformHelper(RewriterBase &rewriter,
                                       linalg::WinogradFilterTransformOp op) {
  Location loc = op.getLoc();
  Value filter = op.getFilter();
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();
  int64_t filterH = filterShape[1];
  int64_t filterW = filterShape[2];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = filterH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = filterW != 1;
  Value transformedFilter =
      filterTransform(rewriter, loc, filter, op.getOutput(), op.getM(),
                      op.getR(), leftTransform, rightTransform);
  if (!transformedFilter)
    return failure();

  rewriter.replaceOp(op, transformedFilter);

  return transformedFilter.getDefiningOp();
}

FailureOr<Operation *>
decomposeWinogradInputTransformHelper(RewriterBase &rewriter,
                                      linalg::WinogradInputTransformOp op) {
  Location loc = op.getLoc();
  Value input = op.getInput();
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();
  int64_t inputH = inputShape[1];
  int64_t inputW = inputShape[2];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = inputH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = inputW != 1;
  Value transformedInput =
      inputTransform(rewriter, loc, op.getInput(), op.getOutput(), op.getM(),
                     op.getR(), leftTransform, rightTransform);
  if (!transformedInput)
    return failure();

  rewriter.replaceOp(op, transformedInput);

  return transformedInput.getDefiningOp();
}

FailureOr<Operation *>
decomposeWinogradOutputTransformHelper(RewriterBase &rewriter,
                                       linalg::WinogradOutputTransformOp op) {
  Location loc = op.getLoc();
  Value value = op.getValue();
  auto valueType = cast<ShapedType>(value.getType());
  auto valueShape = valueType.getShape();
  int64_t valueH = valueShape[0];
  int64_t valueW = valueShape[1];

  // For F(m x 1, r x 1), we only need to do left side transform.
  bool leftTransform = valueH != 1;
  // For F(1 x m, 1 x r), we only need to do right side transform.
  bool rightTransform = valueW != 1;
  Value transformedOutput =
      outputTransform(rewriter, loc, value, op.getOutput(), op.getM(),
                      op.getR(), leftTransform, rightTransform);
  if (!transformedOutput)
    return failure();

  rewriter.replaceOp(op, transformedOutput);

  return transformedOutput.getDefiningOp();
}

class DecomposeWinogradFilterTransform final
    : public OpRewritePattern<linalg::WinogradFilterTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradFilterTransformOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(decomposeWinogradFilterTransformHelper(rewriter, op)))
      return failure();

    return success();
  }
};

class DecomposeWinogradInputTransform final
    : public OpRewritePattern<linalg::WinogradInputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradInputTransformOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(decomposeWinogradInputTransformHelper(rewriter, op)))
      return failure();

    return success();
  }
};

class DecomposeWinogradOutputTransform final
    : public OpRewritePattern<linalg::WinogradOutputTransformOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::WinogradOutputTransformOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(decomposeWinogradOutputTransformHelper(rewriter, op)))
      return failure();

    return success();
  }
};

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

FailureOr<Operation *>
decomposeWinogradFilterTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradFilterTransformOp op) {
  return decomposeWinogradFilterTransformHelper(rewriter, op);
}

FailureOr<Operation *>
decomposeWinogradInputTransformOp(RewriterBase &rewriter,
                                  linalg::WinogradInputTransformOp op) {
  return decomposeWinogradInputTransformHelper(rewriter, op);
}

FailureOr<Operation *>
decomposeWinogradOutputTransformOp(RewriterBase &rewriter,
                                   linalg::WinogradOutputTransformOp op) {
  return decomposeWinogradOutputTransformHelper(rewriter, op);
}

void populateWinogradConv2DPatterns(RewritePatternSet &patterns, int64_t m,
                                    int64_t r) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<WinogradConv2DNhwcFhwc>(context, m, r);
}

void populateDecomposeWinogradOpsPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<DecomposeWinogradFilterTransform>(context);
  patterns.insert<DecomposeWinogradInputTransform>(context);
  patterns.insert<DecomposeWinogradOutputTransform>(context);
}

} // end namespace linalg
} // end namespace mlir
