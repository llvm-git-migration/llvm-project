#include "mlir/CAPI/Presburger.h"
#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include <algorithm>
#include <memory>
#include <vector>
using namespace mlir;
using namespace mlir::presburger;

//===----------------------------------------------------------------------===//
// IntegerRelation creation/destruction and basic metadata operations
//===----------------------------------------------------------------------===//

MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreate(unsigned numReservedInequalities,
                                    unsigned numReservedEqualities,
                                    unsigned numReservedCols) {
  auto space = PresburgerSpace::getRelationSpace();
  IntegerRelation *relation = new IntegerRelation(
      numReservedInequalities, numReservedEqualities, numReservedCols, space);
  return wrap(relation);
}

MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreateFromCoefficients(
    const int64_t *inequalityCoefficients, unsigned numInequalities,
    const int64_t *equalityCoefficients, unsigned numEqualities,
    unsigned numDomainVars, unsigned numRangeVars,
    unsigned numExtraReservedInequalities, unsigned numExtraReservedEqualities,
    unsigned numExtraReservedCols) {
  auto space = PresburgerSpace::getRelationSpace(numDomainVars, numRangeVars);
  IntegerRelation *relation =
      new IntegerRelation(numInequalities + numExtraReservedInequalities,
                          numEqualities + numExtraReservedInequalities,
                          numDomainVars + numRangeVars + 1, space);
  unsigned numCols = numRangeVars + numDomainVars + 1;
  for (const int64_t *rowPtr = inequalityCoefficients;
       rowPtr < inequalityCoefficients + numCols * numInequalities;
       rowPtr += numCols) {
    llvm::ArrayRef<int64_t> coef(rowPtr, rowPtr + numCols);
    relation->addInequality(coef);
  }
  for (const int64_t *rowPtr = equalityCoefficients;
       rowPtr < equalityCoefficients + numCols * numEqualities;
       rowPtr += numCols) {
    llvm::ArrayRef<int64_t> coef(rowPtr, rowPtr + numCols);
    relation->addEquality(coef);
  }
  return wrap(relation);
}

void mlirPresburgerIntegerRelationDestroy(
    MlirPresburgerIntegerRelation relation) {
  if (relation.ptr)
    delete reinterpret_cast<IntegerRelation *>(relation.ptr);
}

//===----------------------------------------------------------------------===//
// IntegerRelation binary operations
//===----------------------------------------------------------------------===//

void mlirPresburgerIntegerRelationAppend(MlirPresburgerIntegerRelation lhs,
                                         MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->append(*unwrap(rhs));
}

MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationIntersect(MlirPresburgerIntegerRelation lhs,
                                       MlirPresburgerIntegerRelation rhs) {
  auto result =
      std::make_unique<IntegerRelation>(unwrap(lhs)->intersect(*(unwrap(rhs))));
  return wrap(result.release());
}

bool mlirPresburgerIntegerRelationIsEqual(MlirPresburgerIntegerRelation lhs,
                                          MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->isEqual(*(unwrap(rhs)));
}

bool mlirPresburgerIntegerRelationIsObviouslyEqual(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->isObviouslyEqual(*(unwrap(rhs)));
}

bool mlirPresburgerIntegerRelationIsSubsetOf(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->isSubsetOf(*(unwrap(rhs)));
}

void mlirPresburgerIntegerRelationCompose(MlirPresburgerIntegerRelation lhs,
                                          MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->compose(*unwrap(rhs));
}

void mlirPresburgerIntegerRelationApplyDomain(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->applyDomain(*unwrap(rhs));
}

void mlirPresburgerIntegerRelationApplyRange(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->applyRange(*unwrap(rhs));
}

void mlirPresburgerIntegerRelationMergeAndCompose(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->mergeAndCompose(*unwrap(rhs));
}

void mlirPresburgerIntegerRelationMergeAndAlignSymbols(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->mergeAndAlignSymbols(*unwrap(rhs));
}

unsigned
mlirPresburgerIntegerRelationMergeLocalVars(MlirPresburgerIntegerRelation lhs,
                                            MlirPresburgerIntegerRelation rhs) {
  return unwrap(lhs)->mergeLocalVars(*unwrap(rhs));
}

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Inspection
//===----------------------------------------------------------------------===//

unsigned mlirPresburgerIntegerRelationNumConstraints(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumConstraints();
}

unsigned mlirPresburgerIntegerRelationNumDomainVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumDomainVars();
}

unsigned mlirPresburgerIntegerRelationNumRangeVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumRangeVars();
}

unsigned mlirPresburgerIntegerRelationNumSymbolVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumSymbolVars();
}

unsigned mlirPresburgerIntegerRelationNumLocalVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumLocalVars();
}

unsigned mlirPresburgerIntegerRelationNumDimVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumDimVars();
}

unsigned mlirPresburgerIntegerRelationNumDimAndSymbolVars(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumDimAndSymbolVars();
}

unsigned
mlirPresburgerIntegerRelationNumVars(MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumVars();
}

unsigned
mlirPresburgerIntegerRelationNumCols(MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumCols();
}

unsigned mlirPresburgerIntegerRelationNumEqualities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumEqualities();
}

unsigned mlirPresburgerIntegerRelationNumInequalities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumInequalities();
}

unsigned mlirPresburgerIntegerRelationNumReservedEqualities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumReservedEqualities();
}

unsigned mlirPresburgerIntegerRelationNumReservedInequalities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumReservedInequalities();
}

MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtEq(MlirPresburgerIntegerRelation relation,
                                  unsigned i, unsigned j) {
  return wrap(&unwrap(relation)->atEq(i, j));
}

int64_t
mlirPresburgerIntegerRelationAtEq64(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col) {
  return unwrap(relation)->atEq64(row, col);
}

MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtIneq(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col) {
  return wrap(&unwrap(relation)->atIneq(row, col));
}

int64_t
mlirPresburgerIntegerRelationAtIneq64(MlirPresburgerIntegerRelation relation,
                                      unsigned row, unsigned col) {
  return unwrap(relation)->atIneq64(row, col);
}

unsigned mlirPresburgerIntegerRelationGetNumVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind) {
  return unwrap(relation)->getNumVarKind(static_cast<VarKind>(kind));
}

unsigned mlirPresburgerIntegerRelationGetVarKindOffset(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind) {
  return unwrap(relation)->getVarKindOffset(static_cast<VarKind>(kind));
}

unsigned mlirPresburgerIntegerRelationGetVarKindEnd(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind) {
  return unwrap(relation)->getVarKindEnd(static_cast<VarKind>(kind));
}

unsigned mlirPresburgerIntegerRelationGetVarKindOverLap(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit) {
  return unwrap(relation)->getVarKindOverlap(static_cast<VarKind>(kind),
                                             varStart, varLimit);
}

MlirPresburgerVariableKind mlirPresburgerIntegerRelationGetVarKindAt(
    MlirPresburgerIntegerRelation relation, unsigned pos) {
  return static_cast<MlirPresburgerVariableKind>(
      unwrap(relation)->getVarKindAt(pos));
}

OptionalInt64 mlirPresburgerIntegerRelationGetConstantBound64(
    MlirPresburgerIntegerRelation relation, MlirPresburgerBoundType type,
    unsigned pos) {
  auto r =
      unwrap(relation)->getConstantBound64(static_cast<BoundType>(type), pos);
  if (r.has_value())
    return OptionalInt64{true, *r};
  return OptionalInt64{false, -1};
}

bool mlirPresburgerIntegerRelationHasOnlyDivLocals(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->hasOnlyDivLocals();
}

bool mlirPresburgerIntegerRelationIsFullDim(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->isFullDim();
}

OptionalOptimum mlirPresburgerIntegerRelationFindIntegerLexMin(
    MlirPresburgerIntegerRelation relation) {
  auto r = unwrap(relation)->findIntegerLexMin();
  if (r.isEmpty()) {
    return OptionalOptimum{MlirPresburgerOptimumKind::Empty,
                           {false, nullptr, 8}};
  }
  if (r.isUnbounded())
    return OptionalOptimum{MlirPresburgerOptimumKind::Unbounded,
                           {false, nullptr, 8}};
  int64_t *integerPoint = std::make_unique<int64_t[]>(8).release();
  std::transform((*r).begin(), (*r).end(), integerPoint, int64fromDynamicAPInt);
  return OptionalOptimum{MlirPresburgerOptimumKind::Bounded,
                         {true, integerPoint, 8}};
}

OptionalVectorInt64 mlirPresburgerIntegerRelationFindIntegerSample(
    MlirPresburgerIntegerRelation relation) {
  auto r = unwrap(relation)->findIntegerSample();
  if (!r.has_value())
    return OptionalVectorInt64{false, nullptr, 8};
  int64_t *integerPoint = std::make_unique<int64_t[]>(8).release();
  std::transform((*r).begin(), (*r).end(), integerPoint, int64fromDynamicAPInt);
  return OptionalVectorInt64{true, integerPoint, 8};
}

OptionalInt64 mlirPresburgerIntegerRelationComputeVolume(
    MlirPresburgerIntegerRelation relation) {
  auto r = unwrap(relation)->computeVolume();
  if (!r.has_value())
    return OptionalInt64{false, -1};
  return OptionalInt64{true, int64fromDynamicAPInt(*r)};
}

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Manipulation
//===----------------------------------------------------------------------===//

unsigned
mlirPresburgerIntegerRelationInsertVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned pos, unsigned num) {
  return unwrap(relation)->insertVar(static_cast<VarKind>(kind), pos, num);
}

unsigned
mlirPresburgerIntegerRelationAppendVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned num) {
  return unwrap(relation)->appendVar(static_cast<VarKind>(kind), num);
}

void mlirPresburgerIntegerRelationAddEquality(
    MlirPresburgerIntegerRelation relation, const std::vector<int64_t> &eq) {
  unwrap(relation)->addEquality(eq);
}

/// Adds an inequality with the given coefficients.
void mlirPresburgerIntegerRelationAddInequality(
    MlirPresburgerIntegerRelation relation, const std::vector<int64_t> &inEq) {
  unwrap(relation)->addInequality(inEq);
}

void mlirPresburgerIntegerRelationEliminateRedundantLocalVar(
    MlirPresburgerIntegerRelation relation, unsigned posA, unsigned posB) {
  return unwrap(relation)->eliminateRedundantLocalVar(posA, posB);
}

void mlirPresburgerIntegerRelationRemoveVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned pos) {
  return unwrap(relation)->removeVar(static_cast<VarKind>(kind), pos);
}

void mlirPresburgerIntegerRelationRemoveVarRangeKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit) {
  return unwrap(relation)->removeVarRange(static_cast<VarKind>(kind), varStart,
                                          varLimit);
}

void mlirPresburgerIntegerRelationRemoveVar(
    MlirPresburgerIntegerRelation relation, unsigned pos) {
  return unwrap(relation)->removeVar(pos);
}

void mlirPresburgerIntegerRelationRemoveEquality(
    MlirPresburgerIntegerRelation relation, unsigned pos) {
  return unwrap(relation)->removeEquality(pos);
}

void mlirPresburgerIntegerRelationRemoveInequality(
    MlirPresburgerIntegerRelation relation, unsigned pos) {
  return unwrap(relation)->removeInequality(pos);
}

void mlirPresburgerIntegerRelationRemoveEqualityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end) {
  return unwrap(relation)->removeEqualityRange(start, end);
}

void mlirPresburgerIntegerRelationRemoveInequalityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end) {
  return unwrap(relation)->removeInequalityRange(start, end);
}

void mlirPresburgerIntegerRelationSwapVar(
    MlirPresburgerIntegerRelation relation, unsigned posA, unsigned posB) {
  return unwrap(relation)->swapVar(posA, posB);
}

void mlirPresburgerIntegerRelationClearConstraints(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->clearConstraints();
}

/// Sets the `values.size()` variables starting at `po`s to the specified
/// values and removes them.
void mlirPresburgerIntegerRelationSetAndEliminate(
    MlirPresburgerIntegerRelation relation, unsigned pos, const int64_t *values,
    unsigned valuesSize) {
  return unwrap(relation)->setAndEliminate(
      pos, ArrayRef<int64_t>(values, values + valuesSize));
}

void mlirPresburgerIntegerRelationRemoveIndependentConstraints(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num) {
  return unwrap(relation)->removeIndependentConstraints(pos, num);
}

bool mlirPresburgerIntegerRelationIsHyperRectangular(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num) {
  return unwrap(relation)->isHyperRectangular(pos, num);
}

void mlirPresburgerIntegerRelationRemoveTrivialRedundancy(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->removeTrivialRedundancy();
}

void mlirPresburgerIntegerRelationRemoveRedundantInequalities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->removeRedundantInequalities();
}

void mlirPresburgerIntegerRelationRemoveRedundantConstraints(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->removeRedundantConstraints();
}

void mlirPresburgerIntegerRelationRemoveDuplicateDivs(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->removeDuplicateDivs();
}

void mlirPresburgerIntegerRelationSimplify(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->simplify();
}

void mlirPresburgerIntegerRelationConvertVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind,
    unsigned pos) {
  return unwrap(relation)->convertVarKind(static_cast<VarKind>(srcKind),
                                          varStart, varLimit,
                                          static_cast<VarKind>(dstKind), pos);
}
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertVarKindNoPos(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind) {
  mlirPresburgerIntegerRelationConvertVarKind(
      relation, srcKind, varStart, varLimit, dstKind,
      mlirPresburgerIntegerRelationGetNumVarKind(relation, dstKind));
}
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertToLocal(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit) {
  mlirPresburgerIntegerRelationConvertVarKindNoPos(
      relation, kind, varStart, varLimit, MlirPresburgerVariableKind::Local);
}

void mlirPresburgerIntegerRelationInverse(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->inverse();
}

void mlirPresburgerIntegerRelationRemoveTrivialEqualities(
    MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->removeTrivialEqualities();
}

//===----------------------------------------------------------------------===//
// IntegerRelation Dump
//===----------------------------------------------------------------------===//
void mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation) {
  unwrap(relation)->dump();
}