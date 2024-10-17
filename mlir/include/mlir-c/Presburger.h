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

enum MlirPresburgerBoundType { EQ, LB, UB };

enum MlirPresburgerOptimumKind { Empty, Unbounded, Bounded };

struct OptionalInt64 {
  bool hasValue;
  int64_t value;
};

typedef struct OptionalInt64 OptionalInt64;

struct OptionalVectorInt64 {
  bool hasValue;
  const int64_t *data;
  int64_t size;
};

typedef struct OptionalVectorInt64 OptionalVectorInt64;

struct OptionalOptimum {
  enum MlirPresburgerOptimumKind kind;
  OptionalVectorInt64 vector;
};

typedef struct OptionalOptimum OptionalOptimum;

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

/// Merge and align symbol variables of `this` and `other` with respect to
/// identifiers. After this operation the symbol variables of both relations
/// have the same identifiers in the same order.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationMergeAndAlignSymbols(
    MlirPresburgerIntegerRelation lhs, MlirPresburgerIntegerRelation rhs);

/// Adds additional local vars to the sets such that they both have the union
/// of the local vars in each set, without changing the set of points that
/// lie in `this` and `other`.
///
/// While taking union, if a local var in `other` has a division
/// representation which is a duplicate of division representation, of another
/// local var, it is not added to the final union of local vars and is instead
/// merged. The new ordering of local vars is:
///
/// [Local vars of `this`] [Non-merged local vars of `other`]
///
/// The relative ordering of local vars is same as before.
///
/// After merging, if the `i^th` local variable in one set has a known
/// division representation, then the `i^th` local variable in the other set
/// either has the same division representation or no known division
/// representation.
///
/// The spaces of both relations should be compatible.
///
/// Returns the number of non-merged local vars of `other`, i.e. the number of
/// locals that have been added to `this`.
MLIR_CAPI_EXPORTED unsigned
mlirPresburgerIntegerRelationMergeLocalVars(MlirPresburgerIntegerRelation lhs,
                                            MlirPresburgerIntegerRelation rhs);

MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationCompose(MlirPresburgerIntegerRelation lhs,
                                     MlirPresburgerIntegerRelation rhs);

MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationApplyDomain(MlirPresburgerIntegerRelation lhs,
                                         MlirPresburgerIntegerRelation rhs);

MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationApplyRange(MlirPresburgerIntegerRelation lhs,
                                        MlirPresburgerIntegerRelation rhs);

/// Given a relation `other: (A -> B)`, this operation merges the symbol and
/// local variables and then takes the composition of `other` on `this: (B ->
/// C)`. The resulting relation represents tuples of the form: `A -> C`.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationMergeAndCompose(MlirPresburgerIntegerRelation lhs,
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

/// Returns the constant bound for the pos^th variable if there is one;
/// std::nullopt otherwise.
MLIR_CAPI_EXPORTED OptionalInt64
mlirPresburgerIntegerRelationGetConstantBound64(
    MlirPresburgerIntegerRelation relation, MlirPresburgerBoundType type,
    unsigned pos);

/// Check whether all local ids have a division representation.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationHasOnlyDivLocals(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED bool
mlirPresburgerIntegerRelationIsFullDim(MlirPresburgerIntegerRelation relation);

/// Find an integer sample point satisfying the constraints using a
/// branch and bound algorithm with generalized basis reduction, with some
/// additional processing using Simplex for unbounded sets.
///
/// Returns an integer sample point if one exists, or an empty Optional
/// otherwise. The returned value also includes values of local ids.
MLIR_CAPI_EXPORTED OptionalVectorInt64
mlirPresburgerIntegerRelationFindIntegerSample(
    MlirPresburgerIntegerRelation relation);

/// Compute an overapproximation of the number of integer points in the
/// relation. Symbol vars currently not supported. If the computed
/// overapproximation is infinite, an empty optional is returned.
MLIR_CAPI_EXPORTED OptionalInt64 mlirPresburgerIntegerRelationComputeVolume(
    MlirPresburgerIntegerRelation relation);

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

MLIR_CAPI_EXPORTED OptionalOptimum
mlirPresburgerIntegerRelationFindIntegerLexMin(
    MlirPresburgerIntegerRelation relation);

/// Swap the posA^th variable with the posB^th variable.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationSwapVar(MlirPresburgerIntegerRelation relation,
                                     unsigned posA, unsigned posB);

/// Removes all equalities and inequalities.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationClearConstraints(
    MlirPresburgerIntegerRelation relation);

/// Sets the `values.size()` variables starting at `po`s to the specified
/// values and removes them.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationSetAndEliminate(
    MlirPresburgerIntegerRelation relation, unsigned pos, const int64_t *values,
    unsigned valuesSize);

/// Removes constraints that are independent of (i.e., do not have a
/// coefficient) variables in the range [pos, pos + num).
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveIndependentConstraints(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num);

/// Returns true if the set can be trivially detected as being
/// hyper-rectangular on the specified contiguous set of variables.
MLIR_CAPI_EXPORTED bool mlirPresburgerIntegerRelationIsHyperRectangular(
    MlirPresburgerIntegerRelation relation, unsigned pos, unsigned num);

/// Removes duplicate constraints, trivially true constraints, and constraints
/// that can be detected as redundant as a result of differing only in their
/// constant term part. A constraint of the form <non-negative constant> >= 0
/// is considered trivially true. This method is a linear time method on the
/// constraints, does a single scan, and updates in place. It also normalizes
/// constraints by their GCD and performs GCD tightening on inequalities.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveTrivialRedundancy(
    MlirPresburgerIntegerRelation relation);

/// A more expensive check than `removeTrivialRedundancy` to detect redundant
/// inequalities.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationRemoveRedundantInequalities(
    MlirPresburgerIntegerRelation relation);

/// Removes redundant constraints using Simplex. Although the algorithm can
/// theoretically take exponential time in the worst case (rare), it is known
/// to perform much better in the average case. If V is the number of vertices
/// in the polytope and C is the number of constraints, the algorithm takes
/// O(VC) time.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveRedundantConstraints(
    MlirPresburgerIntegerRelation relation);

MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveDuplicateDivs(
    MlirPresburgerIntegerRelation relation);

/// Simplify the constraint system by removing canonicalizing constraints and
/// removing redundant constraints.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationSimplify(MlirPresburgerIntegerRelation relation);

/// Converts variables of kind srcKind in the range [varStart, varLimit) to
/// variables of kind dstKind. If `pos` is given, the variables are placed at
/// position `pos` of dstKind, otherwise they are placed after all the other
/// variables of kind dstKind. The internal ordering among the moved variables
/// is preserved.
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertVarKind(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind,
    unsigned pos);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertVarKindNoPos(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind srcKind,
    unsigned varStart, unsigned varLimit, MlirPresburgerVariableKind dstKind);
MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationConvertToLocal(
    MlirPresburgerIntegerRelation relation, MlirPresburgerVariableKind kind,
    unsigned varStart, unsigned varLimit);

MLIR_CAPI_EXPORTED void mlirPresburgerIntegerRelationRemoveTrivialEqualities(
    MlirPresburgerIntegerRelation relation);

/// Invert the relation i.e., swap its domain and range.
///
/// Formally, let the relation `this` be R: A -> B, then this operation
/// modifies R to be B -> A.
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationInverse(MlirPresburgerIntegerRelation relation);

//===----------------------------------------------------------------------===//
// IntegerRelation Dump
//===----------------------------------------------------------------------===//
MLIR_CAPI_EXPORTED void
mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation);
#ifdef __cplusplus
}
#endif
#endif // MLIR_C_PRESBURGER_H