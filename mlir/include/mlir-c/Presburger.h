#ifndef MLIR_C_PRESBURGER_H
#define MLIR_C_PRESBURGER_H
#include "mlir-c/AffineExpr.h"
#include "mlir-c/Support.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

enum MlirPresburgerVariableKind {
  Symbol,
  Local,
  Domain,
  Range,
  SetDim = Range
};

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name
DEFINE_C_API_STRUCT(MlirPresburgerIntegerRelation, void);
DEFINE_C_API_STRUCT(MlirPresburgerDynamicAPInt, const void);
#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// IntegerRelation creation/destruction and basic metadata operations
//===----------------------------------------------------------------------===//

/// Constructs a relation reserving memory for the specified number
/// of constraints and variables.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreate(unsigned numReservedInequalities,
                                    unsigned numReservedEqualities,
                                    unsigned numReservedCols);

/// Constructs an IntegerRelation from a packed 2D matrix of tableau
/// coefficients in row-major order. The first `numDomainVars` columns are
/// considered domain and the remaining `numRangeVars` columns are domain
/// variables.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreateFromCoefficients(
    const int64_t *inequalityCoefficients, unsigned numInequalities,
    const int64_t *equalityCoefficients, unsigned numEqualities,
    unsigned numDomainVars, unsigned numRangeVars,
    unsigned numExtraReservedInequalities = 0,
    unsigned numExtraReservedEqualities = 0, unsigned numExtraReservedCols = 0);

/// Destroys an IntegerRelation.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDestroy(MlirPresburgerIntegerRelation relation);

//===----------------------------------------------------------------------===//
// IntegerRelation binary operations
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAppend(MlirPresburgerIntegerRelation lhs,
                                    MlirPresburgerIntegerRelation rhs);

/// Return the intersection of the two relations.
/// If there are locals, they will be merged.
MLIR_CAPI_EXPORTED MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationIntersect(MlirPresburgerIntegerRelation lhs,
                                       MlirPresburgerIntegerRelation rhs);

/// Return whether `lhs` and `rhs` are equal. This is integer-exact
/// and somewhat expensive, since it uses the integer emptiness check
/// (see IntegerRelation::findIntegerSample()).
MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsEqual(MlirPresburgerIntegerRelation lhs,
                                     MlirPresburgerIntegerRelation rhs);

MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationIsObviouslyEqual(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs);

MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsSubsetOf(MlirPresburgerIntegerRelation lhs,
                                        MlirPresburgerIntegerRelation rhs);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Inspection
//===----------------------------------------------------------------------===//

/// Returns the value at the specified equality row and column.
MLIR_CAPI_EXPORTED MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtEq(unsigned i, unsigned j);

/// The same, but casts to int64_t. This is unsafe and will assert-fail if the
/// value does not fit in an int64_t.
MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtEq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the value at the specified inequality row and column.
MLIR_CAPI_EXPORTED MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtIneq(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col);

MLIR_CAPI_EXPORTED int64_t mlirPresburgerIntegerRelationAtIneq64(
    MlirPresburgerIntegerRelation relation, unsigned row, unsigned col);

/// Returns the number of inequalities and equalities.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumConstraints(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as domain variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumDomainVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as range variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumRangeVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as symbol variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumSymbolVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as local variables.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumLocalVars(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumDimVars(MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumDimAndSymbolVars(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumVars(MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumCols(MlirPresburgerIntegerRelation relation);

/// Returns the number of equality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumEqualities(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of inequality constraints.
MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumInequalities(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationNumReservedEqualities(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationNumReservedInequalities(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetNumVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindOffset(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindEnd(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind);

MLIR_CAPI_EXPORTED unsigned mlirPresburgerIntegerRelationGetVarKindOverLap(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

/// Return the VarKind of the var at the specified position.
MLIR_CAPI_EXPORTED MlirPresburgerVariableKind
mlirPresburgerIntegerRelationGetVarKindAt(
    MlirPresburgerIntegerRelation relation, unsigned pos);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Manipulation
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationInsertVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned pos, unsigned num = 1);

MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationAppendVar(MlirPresburgerIntegerRelation relation,
                                       MlirPresburgerVariableKind kind,
                                       unsigned num = 1);

/// Adds an equality with the given coefficients.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationAddEquality(MlirPresburgerIntegerRelation relation,
                                         const std::vector<int64_t> &eq);

/// Adds an inequality with the given coefficients.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationAddInequality(
    MlirPresburgerIntegerRelation relation, const std::vector<int64_t> &inEq);

MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationEliminateRedundantLocalVar(
    MlirPresburgerIntegerRelation relation, unsigned posA, unsigned posB);

/// Removes variables of the specified kind with the specified pos (or
/// within the specified range) from the system. The specified location is
/// relative to the first variable of the specified kind.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveVarRangeKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

/// Removes the specified variable from the system.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveVar(MlirPresburgerIntegerRelation relation,
                                       unsigned pos);

/// Remove the (in)equalities at specified position.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveEquality(
    MlirPresburgerIntegerRelation relation, unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveInequality(
    MlirPresburgerIntegerRelation relation, unsigned pos);

/// Remove the (in)equalities at positions [start, end).
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveEqualityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveInequalityRange(
    MlirPresburgerIntegerRelation relation, unsigned start, unsigned end);

//===----------------------------------------------------------------------===//
// IntegerRelation Dump
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation);
#ifdef __cplusplus
}
#endif
#endif // MLIR_C_PRESBURGER_H