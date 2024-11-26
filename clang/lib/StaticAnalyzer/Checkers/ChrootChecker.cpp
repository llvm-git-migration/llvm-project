//===-- ChrootChecker.cpp - chroot usage checks ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines chroot checker, which checks improper use of chroot.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

using namespace clang;
using namespace ento;

// enum value that represent the jail state
enum ChrootKind { NO_CHROOT, ROOT_CHANGED, ROOT_CHANGE_FAILED, JAIL_ENTERED };

// Track chroot state changes for success, failure, state change
// and "jail"
REGISTER_TRAIT_WITH_PROGRAMSTATE(ChrootState, ChrootKind)

// Track the call expression to chroot for accurate
// warning messages
REGISTER_TRAIT_WITH_PROGRAMSTATE(ChrootCall, const Expr *)

namespace {

// This checker checks improper use of chroot.
// The state transitions
//
//                          -> ROOT_CHANGE_FAILED
//                          |
// NO_CHROOT ---chroot(path)--> ROOT_CHANGED ---chdir(/) --> JAIL_ENTERED
//                                  |                               |
//         ROOT_CHANGED<--chdir(..)--      JAIL_ENTERED<--chdir(..)--
//                                  |                               |
//                      bug<--foo()--          JAIL_ENTERED<--foo()--
//
class ChrootChecker : public Checker<eval::Call, check::PreCall> {
  // This bug refers to possibly break out of a chroot() jail.
  const BugType BT_BreakJail{this, "Break out of jail"};

  const CallDescription Chroot{CDM::CLibrary, {"chroot"}, 1},
      Chdir{CDM::CLibrary, {"chdir"}, 1};

public:
  ChrootChecker() {}

  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  void evalChroot(const CallEvent &Call, CheckerContext &C) const;
  void evalChdir(const CallEvent &Call, CheckerContext &C) const;

  /// Searches for the ExplodedNode where chroot was called.
  static const ExplodedNode *getAcquisitionSite(const ExplodedNode *N,
                                                CheckerContext &C);
};

bool ChrootChecker::evalCall(const CallEvent &Call, CheckerContext &C) const {
  if (Chroot.matches(Call)) {
    evalChroot(Call, C);
    return true;
  }
  if (Chdir.matches(Call)) {
    evalChdir(Call, C);
    return true;
  }

  return false;
}

void ChrootChecker::evalChroot(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  ProgramStateManager &Mgr = state->getStateManager();
  const TargetInfo &TI = C.getASTContext().getTargetInfo();
  SValBuilder &SVB = C.getSValBuilder();
  BasicValueFactory &BVF = SVB.getBasicValueFactory();
  ConstraintManager &CM = Mgr.getConstraintManager();

  const QualType sIntTy = C.getASTContext().getIntTypeForBitwidth(
      /*DestWidth=*/TI.getIntWidth(), /*Signed=*/true);

  const Expr *ChrootCE = Call.getOriginExpr();
  if (!ChrootCE)
    return;
  const auto *CE = cast<CallExpr>(Call.getOriginExpr());

  const LocationContext *LCtx = C.getLocationContext();
  NonLoc RetVal =
  C.getSValBuilder()
      .conjureSymbolVal(nullptr, ChrootCE, LCtx, sIntTy, C.blockCount())
      .castAs<NonLoc>();

  ProgramStateRef StateChrootFailed, StateChrootSuccess;
  std::tie(StateChrootFailed, StateChrootSuccess) = state->assume(RetVal);

  const llvm::APSInt &Zero = BVF.getValue(0, sIntTy);
  const llvm::APSInt &Minus1 = BVF.getValue(-1, sIntTy);

  if (StateChrootFailed) {
    StateChrootFailed = CM.assumeInclusiveRange(StateChrootFailed, RetVal,
                                                Minus1, Minus1, true);
    StateChrootFailed = StateChrootFailed->set<ChrootState>(ROOT_CHANGE_FAILED);
    StateChrootFailed = StateChrootFailed->set<ChrootCall>(ChrootCE);
    C.addTransition(StateChrootFailed->BindExpr(CE, LCtx, RetVal));
  }

  if (StateChrootSuccess) {
    StateChrootSuccess =
        CM.assumeInclusiveRange(StateChrootSuccess, RetVal, Zero, Zero, true);
    StateChrootSuccess = StateChrootSuccess->set<ChrootState>(ROOT_CHANGED);
    StateChrootSuccess = StateChrootSuccess->set<ChrootCall>(ChrootCE);
    C.addTransition(StateChrootSuccess->BindExpr(CE, LCtx, RetVal));
  }
}

void ChrootChecker::evalChdir(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef state = C.getState();

  // If there are no jail state, just return.
  const ChrootKind k = C.getState()->get<ChrootState>();
  if (!k)
    return;

  // After chdir("/"), enter the jail, set the enum value JAIL_ENTERED.
  const Expr *ArgExpr = Call.getArgExpr(0);
  SVal ArgVal = C.getSVal(ArgExpr);

  if (const MemRegion *R = ArgVal.getAsRegion()) {
    R = R->StripCasts();
    if (const StringRegion* StrRegion= dyn_cast<StringRegion>(R)) {
      const StringLiteral* Str = StrRegion->getStringLiteral();
      if (Str->getString() == "/") {
        state = state->set<ChrootState>(JAIL_ENTERED);
      }
    }
  }

  C.addTransition(state);
}

const ExplodedNode *ChrootChecker::getAcquisitionSite(const ExplodedNode *N,
                                                      CheckerContext &C) {
  ProgramStateRef State = N->getState();
  // When bug type is resource leak, exploded node N may not have state info
  // for leaked file descriptor, but predecessor should have it.
  if (!State->get<ChrootCall>())
    N = N->getFirstPred();

  const ExplodedNode *Pred = N;
  while (N) {
    State = N->getState();
    if (!State->get<ChrootCall>())
      return Pred;
    Pred = N;
    N = N->getFirstPred();
  }

  return nullptr;
}

// Check the jail state before any function call except chroot and chdir().
void ChrootChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {
  // Ignore chroot and chdir.
  if (matchesAny(Call, Chroot, Chdir))
    return;

  // If jail state is ROOT_CHANGED, generate BugReport.
  const ChrootKind k = C.getState()->get<ChrootState>();
  if (k == ROOT_CHANGED) {
    ExplodedNode *Err =
        C.generateNonFatalErrorNode(C.getState(), C.getPredecessor());
    if (!Err)
      return;
    const Expr *ChrootExpr = C.getState()->get<ChrootCall>();

    const ExplodedNode *ChrootCallNode = getAcquisitionSite(Err, C);
    assert(ChrootCallNode && "Could not find place of stream opening.");

    PathDiagnosticLocation LocUsedForUniqueing;
    if (const Stmt *ChrootStmt = ChrootCallNode->getStmtForDiagnostics())
      LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
          ChrootStmt, C.getSourceManager(),
          ChrootCallNode->getLocationContext());

    std::unique_ptr<PathSensitiveBugReport> R =
        std::make_unique<PathSensitiveBugReport>(
            BT_BreakJail, "No call of chdir(\"/\") immediately after chroot",
            Err, LocUsedForUniqueing,
            ChrootCallNode->getLocationContext()->getDecl());

    R->addNote("chroot called here",
               PathDiagnosticLocation::create(ChrootCallNode->getLocation(),
                                              C.getSourceManager()),
               {ChrootExpr->getSourceRange()});

    C.emitReport(std::move(R));
  }
}

} // namespace

void ento::registerChrootChecker(CheckerManager &mgr) {
  mgr.registerChecker<ChrootChecker>();
}

bool ento::shouldRegisterChrootChecker(const CheckerManager &mgr) {
  return true;
}
