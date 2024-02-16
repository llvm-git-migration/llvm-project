//===-- Lower/ClauseProcessor.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_CLAUASEPROCESSOR_H
#define FORTRAN_LOWER_CLAUASEPROCESSOR_H

#include "DirectivesCommon.h"
#include "OpenMPUtils.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> treatIndexAsSection;

namespace Fortran {
namespace lower {
namespace omp {

using DeclareTargetCapturePair =
    std::pair<mlir::omp::DeclareTargetCaptureClause,
              Fortran::semantics::Symbol>;

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::SmallVector<mlir::Value> bounds,
                mlir::SmallVector<mlir::Value> members, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool isVal = false);

void gatherFuncAndVarSyms(
    const Fortran::parser::OmpObjectList &objList,
    mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause);

void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

class ReductionProcessor {
public:
  // TODO: Move this enumeration to the OpenMP dialect
  enum ReductionIdentifier {
    ID,
    USER_DEF_OP,
    ADD,
    SUBTRACT,
    MULTIPLY,
    AND,
    OR,
    EQV,
    NEQV,
    MAX,
    MIN,
    IAND,
    IOR,
    IEOR
  };

  static ReductionIdentifier
  getReductionType(const Fortran::parser::ProcedureDesignator &pd);

  static ReductionIdentifier getReductionType(
      Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp);

  static bool supportedIntrinsicProcReduction(
      const Fortran::parser::ProcedureDesignator &pd);

  static const Fortran::semantics::SourceName
  getRealName(const Fortran::parser::Name *name) {
    return name->symbol->GetUltimate().name();
  }

  static const Fortran::semantics::SourceName
  getRealName(const Fortran::parser::ProcedureDesignator &pd) {
    const auto *name{Fortran::parser::Unwrap<Fortran::parser::Name>(pd)};
    assert(name && "Invalid Reduction Intrinsic.");
    return getRealName(name);
  }

  static std::string getReductionName(llvm::StringRef name, mlir::Type ty) {
    return (llvm::Twine(name) +
            (ty.isIntOrIndex() ? llvm::Twine("_i_") : llvm::Twine("_f_")) +
            llvm::Twine(ty.getIntOrFloatBitWidth()))
        .str();
  }

  static std::string getReductionName(
      Fortran::parser::DefinedOperator::IntrinsicOperator intrinsicOp,
      mlir::Type ty);

  /// This function returns the identity value of the operator \p
  /// reductionOpName. For example:
  ///    0 + x = x,
  ///    1 * x = x
  static int getOperationIdentity(ReductionIdentifier redId,
                                  mlir::Location loc) {
    switch (redId) {
    case ReductionIdentifier::ADD:
    case ReductionIdentifier::OR:
    case ReductionIdentifier::NEQV:
      return 0;
    case ReductionIdentifier::MULTIPLY:
    case ReductionIdentifier::AND:
    case ReductionIdentifier::EQV:
      return 1;
    default:
      TODO(loc, "Reduction of some intrinsic operators is not supported");
    }
  }

  static mlir::Value getReductionInitValue(mlir::Location loc, mlir::Type type,
                                           ReductionIdentifier redId,
                                           fir::FirOpBuilder &builder);

  template <typename FloatOp, typename IntegerOp>
  static mlir::Value getReductionOperation(fir::FirOpBuilder &builder,
                                           mlir::Type type, mlir::Location loc,
                                           mlir::Value op1, mlir::Value op2) {
    assert(type.isIntOrIndexOrFloat() &&
           "only integer and float types are currently supported");
    if (type.isIntOrIndex())
      return builder.create<IntegerOp>(loc, op1, op2);
    return builder.create<FloatOp>(loc, op1, op2);
  }

  static mlir::Value createScalarCombiner(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          ReductionIdentifier redId,
                                          mlir::Type type, mlir::Value op1,
                                          mlir::Value op2);

  /// Creates an OpenMP reduction declaration and inserts it into the provided
  /// symbol table. The declaration has a constant initializer with the neutral
  /// value `initValue`, and the reduction combiner carried over from `reduce`.
  /// TODO: Generalize this for non-integer types, add atomic region.
  static mlir::omp::ReductionDeclareOp createReductionDecl(
      fir::FirOpBuilder &builder, llvm::StringRef reductionOpName,
      const ReductionIdentifier redId, mlir::Type type, mlir::Location loc);

  /// Creates a reduction declaration and associates it with an OpenMP block
  /// directive.
  static void
  addReductionDecl(mlir::Location currentLocation,
                   Fortran::lower::AbstractConverter &converter,
                   const Fortran::parser::OmpReductionClause &reduction,
                   llvm::SmallVectorImpl<mlir::Value> &reductionVars,
                   llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
                   llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                       *reductionSymbols = nullptr);
};

/// Class that handles the processing of OpenMP clauses.
///
/// Its `process<ClauseName>()` methods perform MLIR code generation for their
/// corresponding clause if it is present in the clause list. Otherwise, they
/// will return `false` to signal that the clause was not found.
///
/// The intended use is of this class is to move clause processing outside of
/// construct processing, since the same clauses can appear attached to
/// different constructs and constructs can be combined, so that code
/// duplication is minimized.
///
/// Each construct-lowering function only calls the `process<ClauseName>()`
/// methods that relate to clauses that can impact the lowering of that
/// construct.
class ClauseProcessor {
  using ClauseTy = Fortran::parser::OmpClause;

public:
  ClauseProcessor(Fortran::lower::AbstractConverter &converter,
                  Fortran::semantics::SemanticsContext &semaCtx,
                  const Fortran::parser::OmpClauseList &clauses)
      : converter(converter), semaCtx(semaCtx), clauses(clauses) {}

  // 'Unique' clauses: They can appear at most once in the clause list.
  bool
  processCollapse(mlir::Location currentLocation,
                  Fortran::lower::pft::Evaluation &eval,
                  llvm::SmallVectorImpl<mlir::Value> &lowerBound,
                  llvm::SmallVectorImpl<mlir::Value> &upperBound,
                  llvm::SmallVectorImpl<mlir::Value> &step,
                  llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> &iv,
                  std::size_t &loopVarTypeSize) const;
  bool processDefault() const;
  bool processDevice(Fortran::lower::StatementContext &stmtCtx,
                     mlir::Value &result) const;
  bool processDeviceType(mlir::omp::DeclareTargetDeviceType &result) const;
  bool processFinal(Fortran::lower::StatementContext &stmtCtx,
                    mlir::Value &result) const;
  bool processHint(mlir::IntegerAttr &result) const;
  bool processMergeable(mlir::UnitAttr &result) const;
  bool processNowait(mlir::UnitAttr &result) const;
  bool processNumTeams(Fortran::lower::StatementContext &stmtCtx,
                       mlir::Value &result) const;
  bool processNumThreads(Fortran::lower::StatementContext &stmtCtx,
                         mlir::Value &result) const;
  bool processOrdered(mlir::IntegerAttr &result) const;
  bool processPriority(Fortran::lower::StatementContext &stmtCtx,
                       mlir::Value &result) const;
  bool processProcBind(mlir::omp::ClauseProcBindKindAttr &result) const;
  bool processSafelen(mlir::IntegerAttr &result) const;
  bool processSchedule(mlir::omp::ClauseScheduleKindAttr &valAttr,
                       mlir::omp::ScheduleModifierAttr &modifierAttr,
                       mlir::UnitAttr &simdModifierAttr) const;
  bool processScheduleChunk(Fortran::lower::StatementContext &stmtCtx,
                            mlir::Value &result) const;
  bool processSimdlen(mlir::IntegerAttr &result) const;
  bool processThreadLimit(Fortran::lower::StatementContext &stmtCtx,
                          mlir::Value &result) const;
  bool processUntied(mlir::UnitAttr &result) const;

  // 'Repeatable' clauses: They can appear multiple times in the clause list.
  bool
  processAllocate(llvm::SmallVectorImpl<mlir::Value> &allocatorOperands,
                  llvm::SmallVectorImpl<mlir::Value> &allocateOperands) const;
  bool processCopyin() const;
  bool processDepend(llvm::SmallVectorImpl<mlir::Attribute> &dependTypeOperands,
                     llvm::SmallVectorImpl<mlir::Value> &dependOperands) const;
  bool
  processEnter(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool
  processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
            mlir::Value &result) const;
  bool
  processLink(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;

  // This method is used to process a map clause.
  // The optional parameters - mapSymTypes, mapSymLocs & mapSymbols are used to
  // store the original type, location and Fortran symbol for the map operands.
  // They may be used later on to create the block_arguments for some of the
  // target directives that require it.
  bool processMap(mlir::Location currentLocation,
                  const llvm::omp::Directive &directive,
                  Fortran::lower::StatementContext &stmtCtx,
                  llvm::SmallVectorImpl<mlir::Value> &mapOperands,
                  llvm::SmallVectorImpl<mlir::Type> *mapSymTypes = nullptr,
                  llvm::SmallVectorImpl<mlir::Location> *mapSymLocs = nullptr,
                  llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                      *mapSymbols = nullptr) const;
  bool
  processReduction(mlir::Location currentLocation,
                   llvm::SmallVectorImpl<mlir::Value> &reductionVars,
                   llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
                   llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                       *reductionSymbols = nullptr) const;
  bool processSectionsReduction(mlir::Location currentLocation) const;
  bool processTo(llvm::SmallVectorImpl<DeclareTargetCapturePair> &result) const;
  bool
  processUseDeviceAddr(llvm::SmallVectorImpl<mlir::Value> &operands,
                       llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
                       llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
                       llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                           &useDeviceSymbols) const;
  bool
  processUseDevicePtr(llvm::SmallVectorImpl<mlir::Value> &operands,
                      llvm::SmallVectorImpl<mlir::Type> &useDeviceTypes,
                      llvm::SmallVectorImpl<mlir::Location> &useDeviceLocs,
                      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                          &useDeviceSymbols) const;

  template <typename T>
  bool processMotionClauses(Fortran::lower::StatementContext &stmtCtx,
                            llvm::SmallVectorImpl<mlir::Value> &mapOperands) {
    return findRepeatableClause<T>(
        [&](const T *motionClause, const Fortran::parser::CharBlock &source) {
          mlir::Location clauseLocation = converter.genLocation(source);
          fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

          static_assert(std::is_same_v<T, ClauseProcessor::ClauseTy::To> ||
                        std::is_same_v<T, ClauseProcessor::ClauseTy::From>);

          // TODO Support motion modifiers: present, mapper, iterator.
          constexpr llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
              std::is_same_v<T, ClauseProcessor::ClauseTy::To>
                  ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO
                  : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

          for (const Fortran::parser::OmpObject &ompObject :
               motionClause->v.v) {
            llvm::SmallVector<mlir::Value> bounds;
            std::stringstream asFortran;
            Fortran::lower::AddrAndBoundsInfo info =
                Fortran::lower::gatherDataOperandAddrAndBounds<
                    Fortran::parser::OmpObject, mlir::omp::DataBoundsOp,
                    mlir::omp::DataBoundsType>(
                    converter, firOpBuilder, semaCtx, stmtCtx, ompObject,
                    clauseLocation, asFortran, bounds, treatIndexAsSection);

            auto origSymbol =
                converter.getSymbolAddress(*getOmpObjectSymbol(ompObject));
            mlir::Value symAddr = info.addr;
            if (origSymbol && fir::isTypeWithDescriptor(origSymbol.getType()))
              symAddr = origSymbol;

            // Explicit map captures are captured ByRef by default,
            // optimisation passes may alter this to ByCopy or other capture
            // types to optimise
            mlir::Value mapOp = createMapInfoOp(
                firOpBuilder, clauseLocation, symAddr, mlir::Value{},
                asFortran.str(), bounds, {},
                static_cast<std::underlying_type_t<
                    llvm::omp::OpenMPOffloadMappingFlags>>(mapTypeBits),
                mlir::omp::VariableCaptureKind::ByRef, symAddr.getType());

            mapOperands.push_back(mapOp);
          }
        });
  }

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(mlir::Location currentLocation,
                   llvm::omp::Directive directive) const {
    auto checkUnhandledClause = [&](const auto *x) {
      if (!x)
        return;
      TODO(
          currentLocation,
          "Unhandled clause " +
              llvm::StringRef(Fortran::parser::ParseTreeDumper::GetNodeName(*x))
                  .upper() +
              " in " + llvm::omp::getOpenMPDirectiveName(directive).upper() +
              " construct");
    };

    for (ClauseIterator it = clauses.v.begin(); it != clauses.v.end(); ++it)
      (checkUnhandledClause(std::get_if<Ts>(&it->u)), ...);
  }

private:
  using ClauseIterator = std::list<ClauseTy>::const_iterator;

  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end) {
    for (ClauseIterator it = begin; it != end; ++it) {
      if (std::get_if<T>(&it->u))
        return it;
    }

    return end;
  }

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  const T *
  findUniqueClause(const Fortran::parser::CharBlock **source = nullptr) const {
    ClauseIterator it = findClause<T>(clauses.v.begin(), clauses.v.end());
    if (it != clauses.v.end()) {
      if (source)
        *source = &it->source;
      return &std::get<T>(it->u);
    }
    return nullptr;
  }

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  bool findRepeatableClause(
      std::function<void(const T *, const Fortran::parser::CharBlock &source)>
          callbackFn) const {
    bool found = false;
    ClauseIterator nextIt, endIt = clauses.v.end();
    for (ClauseIterator it = clauses.v.begin(); it != endIt; it = nextIt) {
      nextIt = findClause<T>(it, endIt);

      if (nextIt != endIt) {
        callbackFn(&std::get<T>(nextIt->u), nextIt->source);
        found = true;
        ++nextIt;
      }
    }
    return found;
  }

  /// Set the `result` to a new `mlir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(mlir::UnitAttr &result) const {
    if (findUniqueClause<T>()) {
      result = converter.getFirOpBuilder().getUnitAttr();
      return true;
    }
    return false;
  }

  Fortran::lower::AbstractConverter &converter;
  Fortran::semantics::SemanticsContext &semaCtx;
  const Fortran::parser::OmpClauseList &clauses;
};

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CLAUASEPROCESSOR_H
