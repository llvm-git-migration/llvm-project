//===- TosaRemoveRedundantTransposes.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ----------
// Motivation:
// ----------

// Some legalization pathways introduce redundant tosa.TRANSPOSE
// operations that result in avoidable data movement. For example,
// PyTorch -> TOSA contains a lot of unnecessary transposes due
// to conversions between NCHW and NHWC.

// We wish to remove all the ones that we can, since in general
// it is possible to remove upwards of 90% of these transposes
// in a provable manner.
//
// -------------------
// High-Level Overview:
// -------------------

// The pass begins at a downstream transpose with some perms tensor.
// It traverses the dependencies upward, accepting only TosaElementwise
// operators. Dependencies must terminate in nullifying transposes (when
// composed, they form the identity), reshapes, or consts.

// Conceptually, we then "bubble up" the downstream transpose until
// we hit the sources. For constants, we generate a new constants, composed
// with the downstream transpose. For nullifying transposes, we "cancel"
// them. For reshapes, we generally cannot "bubble" through them, so we
// insert the downstream transpose there.

// We then ensure that we do not cause any duplication by replacing usages
// of the downstream transpose with the converted value of the operand
// that feeds into it (after this bubble-up process). We do this by analyzing
// the dependency fan-ins across all transposes with the same perms tensor
// in order to ensure that they do not have uses outside this group, which
// would cause the old code section to remain "live", and not removed by
// canonicalization.

// --------------
// Impact of Pass:
// --------------

// For the ResNet18 network, we are able to reduce it to 5 transposes, from
// 56 -- with the patching of the torch dense_resource artifacts with dense
// attributes. Otherwise, without that patch, we reduce to 23, since we cannot
// fold those artifacts.

// In the second case (56 -> 23), instruction count is reduced by exactly 33.
// There are 3 transposes that would be removed if we omitted the fan-in
// analysis, however, with fan-in analysis, we end up with ~15 less operations,
// due to the lack of duplication.

// For ResNet50, the results are essentially identical.

// For MobilenetV3, we reduce the number of transposes from 82 to 38 without
// taking care of upstream constants. After also taking care of constants, we
// reduce it to 20 transposes. The remaining have a use elsewhere outside
// of the fan-in cones. The pass alone (after --canonicalize is run on the
// initial network), is responsible for the removal of 48 of the transposes.

// Due to cases where a constant is used purely in its NCHW form without a
// transpose to NHWC and  also separately used in a place where the downstream
// converts to NHWC, we do end up with 7 additional constants; however, due to
// their small size, this has minimal memory footprint.

// -----------
// Future Work:
// -----------

// (1)

// Evaluate tradeoffs with the duplication of ConstOp, especially
// across many downstream transposes with different perms, which can result
// in the same ConstOp being duplicated (but transposed) multiple times.

// Observe tradeoffs between a lower memory footprint and potentially
// converting many fan-ins of downstream transposes with the same perms,
// which if not converted may affect ability of other inter-dependent fan-in
// to convert.

// (2)

// Restrict the propagation of transposes up their fan-in cone if one
// of the sources is a ReshapeOp for which the inserted TransposeOp would
// not be a TransposeOp that lends itself to the TransposeIsReshape
// Canonicalization, which permits them to be folded to a single ReshapeOp.

// Observe impact on how this restriction may be detrimental to the
// conversion of other downstream transpose conversions due to the
// fan-in cone analysis. Additionally, consider cases where there
// may be multiple upstream transposes that could be removed as a
// result of this -- and trade that off with how many you would
// effectively insert if the ReshapeOp/TransposeOp can't be folded
// to a single ReshapeOp.

// (3)

// Make the pass more general, beyond just allowing upstream transposes
// to be nullifying. For example,

// transpose1 -> ... -> transpose2

// where transpose2(transpose1) do not cancel to identity.

// This can be done by propagating the downstream transpose up
// and inserting after transpose1, just like how it is done for
// reshape. However, in the case of chains like

// transpose1 -> ... -> transpose2 -> ... -> transpose3

// this could require running the current runOnOperation() function
// until we converge. This can be done by stopping when all transposes
// that we can successfully collect the fan-ins of have the owner
// of their first operand being either another TransposeOp or a
// ReshapeOp, since those are what we propagate to and where we leave
// behind / insert another TransposeOp. Otherwise, we would could potentially
// have infinite looping.

// This additionally has the implication that we would not replace any
// transposes and instead we could have canonicalization handle that.

// (4)

// Add support for more instructions (for example, those that reduce
// alongside an axis) to be one of the intervening operations in the
// fan-in cones (other than those with TosaElementwiseOperator trait).

// (5)

// Support bubbling transposes up to the input parameter. May not
// need extensive fan-in analysis as no operation cost associated
// if used elsewhere.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include <memory>
#include <set>
#include <stack>

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAREMOVEREDUNDANTTRANSPOSES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// TOSA Remove Redundant Transposes Pass.
//===----------------------------------------------------------------------===//

namespace {

struct TosaRemoveRedundantTransposes final
    : public tosa::impl::TosaRemoveRedundantTransposesBase<
          TosaRemoveRedundantTransposes> {
  void runOnOperation() override;

private:
  // This will collect all the data dependencies for the given Operation
  // up to and including ConstOp, ReshapeOp, and TransposeOp.
  bool collectFanIn(Operation *op, SetVector<Operation *> &collected);
  bool convertDependentOps(SetVector<Operation *> &dependentOps,
                           DenseMap<Value, Value> &valuesMap,
                           IRRewriter &rewriter,
                           ArrayRef<int32_t> downstreamPerms);

  // Checks if the two permutations, when applied consecutively, result
  // in the identity.
  bool areNullifyingTransposes(ArrayRef<int32_t> perms1,
                               ArrayRef<int32_t> perms2);

  // This is meant to apply to operations with the TosaElementwiseOperator
  // trait.
  std::optional<Value>
  buildMappedToValue(Operation *op, const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms);

  // This updates valuesMap when we encounter another TransposeOp as a
  // dependency of the downstream one. %0 = tosa.transpose %arg0 <- applies to
  // this %1 = tosa.transpose %0 <- when tracking back from this
  std::optional<Value>
  buildMappedToValue(tosa::TransposeOp transposeOp,
                     const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms);

  // Inserts the downstream TransposeOp after the ReshapeOp, since we generally
  // cannot propagate through it.
  std::optional<Value>
  buildMappedToValue(tosa::ReshapeOp reshapeOp,
                     const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms);

  // We may have something like:
  // %0 = tosa.const
  // %1 = tosa.transpose
  // %2 = tosa.add %0, %1
  // %3 = tosa.transpose %2
  // that --tosa-layerwise-const-fold wouldn't handle. This use shows up
  // in MobilenetV3.
  std::optional<Value>
  buildMappedToValue(tosa::ConstOp constOp,
                     const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms);

  // Checks which TransposeOp we should "replace", turning their converted
  // chains of ops, through which they were propagated, "live", and the old code
  // "dead." Attempts to avoid doing so when doing so would result in the old
  // code staying "live," resulting in duplication. Relies on --canonicalize to
  // remove the dead code that results from performing said replacement.
  std::set<tosa::TransposeOp> getGoodReplacements(
      ArrayRef<int32_t> perms,
      std::vector<std::pair<tosa::TransposeOp, SetVector<Operation *>>>
          &transposeInfo);

  // Helper function for getGoodReplacements to check if some TransposeOp's
  // dependencies are OK.
  bool dependenciesAreValid(
      ArrayRef<int32_t> perms, const SetVector<Operation *> &dependentOps,
      std::set<tosa::TransposeOp> &validTransposes,
      std::vector<std::pair<tosa::TransposeOp, SetVector<Operation *>>>
          &transposeInfo);

  // Applies perms to the DenseElementsAttr.
  // If it returns std::nullopt, it also triggers pass failure, since verifier
  // guarantees from TOSA are not in place (and otherwise, if used elsewhere
  // it should fail).
  // This is a basic API and may benefit from refactor into the core MLIR APIs.
  std::optional<DenseElementsAttr>
  transposeDenseAttribute(DenseElementsAttr input, ArrayRef<int32_t> perms);
};

std::optional<DenseElementsAttr>
TosaRemoveRedundantTransposes::transposeDenseAttribute(
    DenseElementsAttr input, ArrayRef<int32_t> perms) {
  RankedTensorType oldType = llvm::cast<RankedTensorType>(input.getType());
  RankedTensorType newType = RankedTensorType::get(
      tosa::applyTOSAPermutation(oldType.getShape(), perms),
      oldType.getElementType());
  size_t rank = oldType.getRank();

  if (input.isSplat())
    return input.reshape(newType);
  // Asserted by TransposeOp verifier and TOSA disallowing tensor with dimension
  // 0.
  // If not in place, something is very wrong.
  if (rank <= 0 || oldType.getNumElements() <= 0 || perms.size() != rank) {
    signalPassFailure();
    return std::nullopt;
  }

  // The algorithm is approximately as follows:
  // input: perms, input flat array, input tensor type
  // (1/2) determine the strides of input/output if
  // they were strided in row-major order. (3) adjust the strides for the
  // input to be in the same order of indices as the output is written.
  // (4) process dimension by dimension. example: perms 2, 0, 1; input
  // 2x3x4; output 4x2x3 for i ... 4, j ... 2, k ... 3: output[i][j][k] =
  // input[j][k][i] output[6i + 3j + k] = input[12j + 4k + i] and we adjust
  // input strides to be as input[i + 12j + 4k] so we may process
  // layer-by-layer.

  // Step 1/2: Strides for input. We ignore output since row-major and can just
  // push_back.

  SmallVector<int64_t> originalInputStrides(rank);
  originalInputStrides[rank - 1] = 1;
  // index with int64_t to avoid overflow
  for (int64_t i = rank - 2; i >= 0; i--)
    originalInputStrides[i] =
        originalInputStrides[i + 1] * oldType.getDimSize(i + 1);

  // Step 3: Transpose strides of input to be same indexing (i, j, k, ...) as
  // output which is done in row-major order.

  SmallVector<int64_t> newInputStrides;
  newInputStrides.reserve(rank);
  for (int32_t v : perms)
    newInputStrides.push_back(originalInputStrides[v]);

  // Step 4: Write out the transposed "flat array" dimension by dimension.

  auto inputArray = input.getValues<Attribute>();
  SmallVector<std::pair<int64_t, int64_t>> boundsAndStrides;
  for (size_t i = 0; i < rank; i++)
    boundsAndStrides.push_back({newType.getDimSize(i), newInputStrides[i]});

  SmallVector<Attribute> resultArray;
  resultArray.reserve(inputArray.size());

  std::function<void(int64_t,
                     SmallVector<std::pair<int64_t, int64_t>>::const_iterator)>
      processTransposeDim = [&](auto accumulatedIndex, auto it) {
        if (it == boundsAndStrides.end()) {
          resultArray.push_back(inputArray[accumulatedIndex]);
          return;
        }

        for (int64_t i = 0; i < it->first; i++) {
          int64_t j = accumulatedIndex + i * it->second;
          processTransposeDim(j, it + 1);
        }
      };

  processTransposeDim(0, boundsAndStrides.begin());

  return DenseElementsAttr::get(newType, resultArray);
}

// The SetVector should only contain ConstOp, ReshapeOp, TransposeOp
// as the sources of the data dependencies, and TosaElementWiseOperator
// after that, if the function returns true.
bool TosaRemoveRedundantTransposes::collectFanIn(
    Operation *op, SetVector<Operation *> &collected) {
  // Can occur if defined through the parameter to a func.func.
  if (!op)
    return false;

  if (!llvm::isa_and_present<tosa::TosaDialect>(op->getDialect()))
    return false;

  // Prevent extra work if already seen.
  if (collected.contains(op))
    return true;

  // Throw it out so later don't have to deal with this.
  if (op->getNumResults() != 1 ||
      !llvm::isa<RankedTensorType>(op->getResult(0).getType()))
    return false;

  // We don't wish to traverse up a ReshapeOp,
  // since generally we can't propagate a TransposeOp through it.
  // TransposeOp, ReshapeOp, ConstOp will have no in-edges in the data
  // dependency graph we construct for the downstream TransposeOp.
  if (!llvm::isa<tosa::TransposeOp>(op) && !llvm::isa<tosa::ReshapeOp>(op) &&
      !llvm::isa<tosa::ConstOp>(op)) {

    if (!op->hasTrait<OpTrait::tosa::TosaElementwiseOperator>())
      return false;

    for (Value operand : op->getOperands()) {

      if (!collectFanIn(operand.getDefiningOp(), collected))
        return false;
    }
  }

  // Insert in topological order.
  collected.insert(op);

  return true;
}

// Assuming that due to the verification of TransposeOp
// perms arrays are permutations of 0 - perms.size() - 1.
bool TosaRemoveRedundantTransposes::areNullifyingTransposes(
    ArrayRef<int32_t> perms1, ArrayRef<int32_t> perms2) {
  if (perms1.size() != perms2.size())
    return false;
  for (int32_t i = 0; i < static_cast<int32_t>(perms1.size()); i++)
    if (perms2[perms1[i]] != i)
      return false;
  return true;
}

// Primarily overload for those with TosaElementwiseOperator trait.
// The other ones handle the case of the operations that occur at the
// roots of the data dependency graph (ConstOp, ReshapeOp, TransposeOp).
std::optional<Value> TosaRemoveRedundantTransposes::buildMappedToValue(
    Operation *op, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms) {
  if (op->getNumResults() != 1 ||
      !op->hasTrait<OpTrait::tosa::TosaElementwiseOperator>())
    return std::nullopt;

  auto resultType = llvm::cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<Value, 3> operands;
  for (Value v : op->getOperands()) {
    if (valuesMap.contains(v)) {
      operands.push_back(valuesMap.at(v));
    } else {
      return std::nullopt;
    }
  }

  // Conceptually, we propagate the downstream TransposeOp through
  // these interveaning operations. For example,
  // %0 = tosa.clamp %input : (tensor<2x3xi32>) -> tensor<2x3xi32>
  // %1 = tosa.transpose %0 {perms = [1, 0]} : (tensor<2x3xi32>) ->
  // tensor<3x2xi32> becomes: %0 = tosa.transpose %input {perms = [1, 0]} :
  // (tensor<2x3xi32>) -> tensor<3x2xi32> %1 = tosa.clamp %0 : (tensor<3x2xi32>)
  // -> tensor<3x2xi32>) We construct this new tosa.clamp here, but it doesn't
  // turn "live" until the final downstream transpose in the chain (that we are
  // currently traversing up its dependencies) is replaced with the proper value
  // from this new chain.
  return rewriter
      .create(op->getLoc(),
              rewriter.getStringAttr(op->getName().getStringRef()), operands,
              RankedTensorType::get(tosa::applyTOSAPermutation(
                                        resultType.getShape(), downstreamPerms),
                                    resultType.getElementType()),
              op->getAttrs())
      ->getResult(0);
}

std::optional<Value> TosaRemoveRedundantTransposes::buildMappedToValue(
    tosa::TransposeOp transposeOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms) {
  SmallVector<int32_t> perms;
  if (failed(transposeOp.getConstantPerms(perms)) ||
      !areNullifyingTransposes(downstreamPerms, perms))
    return std::nullopt;
  return transposeOp.getInput1();
}

std::optional<Value> TosaRemoveRedundantTransposes::buildMappedToValue(
    tosa::ReshapeOp reshapeOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms) {
  auto reshapeOutput = reshapeOp.getOutput();
  auto reshapeOutputType =
      llvm::cast<RankedTensorType>(reshapeOutput.getType());
  if (downstreamPerms.size() !=
      static_cast<size_t>(reshapeOutputType.getRank()))
    return std::nullopt;

  // Since perms is guaranteed to be i32,
  // then this is OK, since --canonicalize can fold them into one.
  auto permsAttr = rewriter.getI32TensorAttr(downstreamPerms);
  auto permsValue = rewriter.create<tosa::ConstOp>(
      reshapeOp.getLoc(), permsAttr.getType(), permsAttr);

  // We cannot propagate the TransposeOp through the ReshapeOp, like
  // we do with those with TosaElementwiseOperator attribute.
  // In general, there won't be any transpose upstream of the ReshapeOp,
  // such as in the ResNet networks.

  // By propagating it here, we permit ourselves to allow this dependency
  // chain to be removed, and also potentially later remove this one
  // if the inserted TransposeOp lends itself to the TransposeIsReshape
  // canonicalization. For example, in the common PyTorch networks.

  // There can be pathological behavior if there are many TransposeOp
  // that do not lend themselves to the TransposeIsReshape canonicalization.
  auto insertedOp = rewriter.create<tosa::TransposeOp>(
      reshapeOp.getLoc(),
      RankedTensorType::get(tosa::applyTOSAPermutation(
                                reshapeOutputType.getShape(), downstreamPerms),
                            reshapeOutputType.getElementType()),
      reshapeOutput, permsValue->getResult(0));
  return insertedOp->getResult(0);
}

std::optional<Value> TosaRemoveRedundantTransposes::buildMappedToValue(
    tosa::ConstOp constOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms) {
  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr)
    return std::nullopt;
  auto maybeNewDenseAttr = transposeDenseAttribute(denseAttr, downstreamPerms);
  if (!maybeNewDenseAttr.has_value())
    return std::nullopt;
  auto newDenseAttr = maybeNewDenseAttr.value();
  auto newConstOp = rewriter.create<tosa::ConstOp>(
      constOp.getLoc(), newDenseAttr.getType(), newDenseAttr);
  return newConstOp->getResult(0);
}

bool TosaRemoveRedundantTransposes::convertDependentOps(
    SetVector<Operation *> &dependentOps, DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> downstreamPerms) {

  for (Operation *op : dependentOps) {
    if (!op || op->getNumResults() != 1)
      return false;

    Value priorValue = op->getResult(0);

    // It's possible on a prior transposeOp
    // we had the same dependency and already resolved it.
    if (valuesMap.contains(priorValue))
      continue;

    // Keep converted ops close to the original.
    rewriter.setInsertionPointAfter(op);

    std::optional<Value> maybeValue =
        llvm::TypeSwitch<Operation *, std::optional<Value>>(op)
            .Case<tosa::TransposeOp>([&](tosa::TransposeOp transposeOp) {
              return buildMappedToValue(transposeOp, valuesMap, rewriter,
                                        downstreamPerms);
            })
            .Case<tosa::ReshapeOp>([&](tosa::ReshapeOp reshapeOp) {
              return buildMappedToValue(reshapeOp, valuesMap, rewriter,
                                        downstreamPerms);
            })
            .Case<tosa::ConstOp>([&](tosa::ConstOp constOp) {
              return buildMappedToValue(constOp, valuesMap, rewriter,
                                        downstreamPerms);
            })
            .Default([&](Operation *op) {
              return buildMappedToValue(op, valuesMap, rewriter,
                                        downstreamPerms);
            });

    if (!maybeValue.has_value())
      return false;

    valuesMap[priorValue] = maybeValue.value();
  }

  return true;
}

// Dependencies are valid for an operation if none of them occur outside
// of the proper fan-in cones of the downstream TransposeOp with the same perms
// that we can replace. Described in more detail within.
bool TosaRemoveRedundantTransposes::dependenciesAreValid(
    ArrayRef<int32_t> perms, const SetVector<Operation *> &dependentOps,
    std::set<tosa::TransposeOp> &validTransposes,
    std::vector<std::pair<tosa::TransposeOp, SetVector<Operation *>>>
        &transposeInfo) {
  for (Operation *op : dependentOps) {

    // It's OK wherever ConstOp has uses -- in the worst case, we duplicate.
    // This can be changed later if we find the memory impact is too high.
    if (llvm::isa<tosa::ConstOp>(op))
      continue;

    for (OpOperand &use : op->getUses()) {
      // Want the uses to be (1) contained in the dependentOps of other
      // validTransposes, or (2) to be directly used in a TransposeOp with the
      // same perms. For (2), it is either (a) we inserted this for
      // ReshapeOp conversion or (b) the fan-in is a subset of our
      // dependentOps, so it is also a validTranspose that will eventually be
      // replaced.
      Operation *user = use.getOwner();
      if (auto otherTranspose = llvm::dyn_cast<tosa::TransposeOp>(user)) {
        SmallVector<int32_t> otherPerms;

        // Can later think about cases where transpose -> transpose
        // or reshape -> transpose, where the transposes are not necessarily
        // the same perms as the downstream, if implementing a more general
        // transform. These could be permitted.
        if (failed(otherTranspose.getConstantPerms(otherPerms)) ||
            !llvm::equal(perms, otherPerms))
          return false;

      } else if (llvm::none_of(
                     transposeInfo,
                     [&validTransposes,
                      user](const std::pair<tosa::TransposeOp,
                                            SetVector<Operation *>> &info) {
                       const auto &[transposeOp, dependentOps] = info;
                       return validTransposes.count(transposeOp) &&
                              dependentOps.contains(user);
                     })) {
        return false;
      }
    }
  }

  return true;
}

// Getting the set of TransposeOp that we can replace without causing
// the old fan-in cones of any TransposeOp to remain "live", i.e, -- not being
// dead code. This is done by iterating the set until convergence, since
// if you are used outside your own fan-in cone, it's possible to be used
// in another fan-in cone of a TransposeOp that is being replaced -- unless
// we find that that one has a usage outside of it too.
std::set<tosa::TransposeOp> TosaRemoveRedundantTransposes::getGoodReplacements(
    ArrayRef<int32_t> perms,
    std::vector<std::pair<tosa::TransposeOp, SetVector<Operation *>>>
        &transposeInfo) {
  // Initially, we assume they are all good to replace,
  // and we whittle them down based on our criteria.
  std::set<tosa::TransposeOp> ableToReplace;
  for (const auto &[transposeOp, _] : transposeInfo)
    ableToReplace.insert(transposeOp);

  bool gotRid;
  do {
    gotRid = false;
    for (const auto &[transposeOp, dependentOps] : transposeInfo) {
      // We don't care about it. Already invalidated.
      if (!ableToReplace.count(transposeOp))
        continue;

      // Check for validity.
      if (!dependenciesAreValid(perms, dependentOps, ableToReplace,
                                transposeInfo)) {
        ableToReplace.erase(transposeOp);
        gotRid = true;
        break;
      }
    }

  } while (gotRid);

  return ableToReplace;
}

void TosaRemoveRedundantTransposes::runOnOperation() {
  // We want to operate only within a single block.
  // Call --inline before to run the pass.
  // This assumption is not strict and could potentially be made more
  // flexible.
  if (!getOperation().getRegion().hasOneBlock())
    return;

  IRRewriter rewriter(&getContext());
  // For each perms, maintain a mapping for converted ops, avoid duplication.
  DenseMap<ArrayRef<int32_t>, DenseMap<Value, Value>> permsToValues;
  // For each perms, we keep track of which tosa::TransposeOp are eligible
  // for replacement alongside their dependentOps.
  DenseMap<ArrayRef<int32_t>,
           std::vector<std::pair<tosa::TransposeOp, SetVector<Operation *>>>>
      permsToTransposeInfo;

  // Necessary for lifetime, since DenseMap keeps a copy of the ArrayRef.
  // Use SmallVector for perms (common-case is <= 4) but std::vector otherwise
  // since no guarantee of smallness.
  std::vector<SmallVector<int32_t>> collectedPerms;

  // This keeps track of the order across all eligible-for-replacement
  // TransposeOp and their perms, a necessity for the final replacements.
  std::stack<std::pair<tosa::TransposeOp, ArrayRef<int32_t>>>
      totalTransposeOrder;

  // We want to reserve the space up front,
  // since SmallVector stores some data internally
  // and the ArrayRef can reference that, which we don't want to get
  // invalidated.
  size_t expectedMaxPerms = 0;
  getOperation().walk([&](tosa::TransposeOp) { expectedMaxPerms += 1; });
  collectedPerms.reserve(expectedMaxPerms);

  getOperation().walk([&](tosa::TransposeOp transposeOp) {
    SetVector<Operation *> dependentOps;
    collectedPerms.emplace_back();
    SmallVector<int32_t> &perms = collectedPerms.back();

    // Dynamic shapes are OK,
    // but the incompatible ones will be rejected later.
    auto input = transposeOp.getInput1();
    auto output = transposeOp.getOutput();

    // However, we don't support unranked tensors.
    if (!llvm::isa<RankedTensorType>(input.getType()) ||
        !llvm::isa<RankedTensorType>(output.getType()))
      return;

    // No transformation when transpose permutation non-constant.
    if (failed(transposeOp.getConstantPerms(perms)))
      return;

    // We let --canonicalize deal with identity transpose.
    if (llvm::equal(llvm::seq<int32_t>(0, perms.size()), perms))
      return;

    // Can fail if some set of basic invariants is not met that we want to
    // perform our conversions.
    if (!collectFanIn(input.getDefiningOp(), dependentOps))
      return;

    // Want to associate valuesMap for already converted of the same perms,
    // since it's possible multiple downstream transposes w/ different perms
    // converge on an op, which would result in different transformations.
    DenseMap<Value, Value> &valuesMap = permsToValues[perms];

    // Attempt to perform the conversions and placements into IR
    // without turning inserted code "live". Also fills out valuesMap.
    // Fails if there is an intermediary we do not support.
    if (!convertDependentOps(dependentOps, valuesMap, rewriter, perms))
      // Some additional operations may have been inserted, but will be
      // removed by dead code elimination and --canonicalize.
      return;

    // This should not happen. If it does -- it's unexpected,
    // so we fail the pass.
    if (!valuesMap.contains(input))
      return signalPassFailure();

    // It's possible the types are not compatible (because of dynamic shapes),
    // and in these cases, want to resolve dynamic shapes before running the
    // pass.
    if (output.getType() != valuesMap.at(input).getType())
      return;

    auto &transposeInfo = permsToTransposeInfo[perms];

    // In general, we might also want to introduce "newDependentOps"
    // if there are new usages that don't fall inside the original fan-ins
    // (like the tosa::TransposeOp we insert for tosa::ReshapeOp),
    // but in this case, that is specialized enough and overlaps
    // with another direct-use tosa::TransposeOp case we need to cover anyway.
    transposeInfo.push_back({transposeOp, dependentOps});

    // This is for the final replacement across all transposes.
    totalTransposeOrder.push({transposeOp, perms});
  });

  // We want to do a full fan-in analysis on a perms-level,
  // since if we do it on a multi-perms level, and they share (due to a shared
  // dependency on a Reshape) then we would also get duplicate ops.
  // Const is special cased.
  std::set<tosa::TransposeOp> ableToReplace;
  for (auto &[perms, transposeInfo] : permsToTransposeInfo) {
    // Gives us back replacements that would never result in any duplicate
    // operations being inserted by us in the IR (i.e, our goal is only to
    // remove transposes, and not create a "new chain" to do so, but replace
    // the existing chains).
    // Ideally, --canonicalize is run before this pass, since it helps this
    // analysis by removing dead code to allow more potentially acceptable
    // transformations.
    auto goodReplacementsForPerms = getGoodReplacements(perms, transposeInfo);
    ableToReplace.insert(goodReplacementsForPerms.begin(),
                         goodReplacementsForPerms.end());
  }

  // We want to do replacement across all transposes
  // in reverse order, due to invalidation of valuesMap mappings
  // if we did it otherwise.
  while (!totalTransposeOrder.empty()) {
    auto [transposeOp, perms] = totalTransposeOrder.top();
    totalTransposeOrder.pop();

    if (ableToReplace.count(transposeOp) == 0)
      continue;

    auto &valuesMap = permsToValues[perms];
    auto input = transposeOp.getInput1();

    // The purpose of this reverse iteration
    // is to avoid valuesMap invalidation. If it happens,
    // something is wrong.
    if (!valuesMap.contains(input))
      return signalPassFailure();

    rewriter.replaceOp(transposeOp, valuesMap.at(input));
  }
}

} // namespace

std::unique_ptr<Pass> tosa::createTosaRemoveRedundantTransposes() {
  return std::make_unique<TosaRemoveRedundantTransposes>();
}
