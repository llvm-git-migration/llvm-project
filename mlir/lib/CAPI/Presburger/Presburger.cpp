#include "mlir/CAPI/Presburger.h"
#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
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

MlirPresburgerVariableKind mlirPresburgerIntegerRelationGetVarKindAt(
    MlirPresburgerIntegerRelation relation, unsigned pos) {
  return wrap(unwrap(relation)->getVarKindAt(pos));
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

void mlirPresburgerIntegerRelationAddEquality(
    MlirPresburgerIntegerRelation relation, const int64_t *coefficients,
    size_t coefficientsSize) {
  unwrap(relation)->addEquality(
      llvm::ArrayRef<int64_t>(coefficients, coefficients + coefficientsSize));
}

void mlirPresburgerIntegerRelationAddInequality(
    MlirPresburgerIntegerRelation relation, const int64_t *coefficients,
    size_t coefficientsSize) {
  unwrap(relation)->addEquality(
      llvm::ArrayRef<int64_t>(coefficients, coefficients + coefficientsSize));
}

void mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation) {
  unwrap(relation)->dump();
}