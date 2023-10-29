//===--- RedundantCastingCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantCastingCheck.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static bool areTypesEquals(QualType S, QualType D) {
  if (S == D)
    return true;

  const auto *TS = S->getAs<TypedefType>();
  const auto *TD = D->getAs<TypedefType>();
  if (TS != TD)
    return false;

  QualType PtrS = S->getPointeeType();
  QualType PtrD = D->getPointeeType();

  if (!PtrS.isNull() && !PtrD.isNull())
    return areTypesEquals(PtrS, PtrD);

  const DeducedType *DT = S->getContainedDeducedType();
  if (DT && DT->isDeduced())
    return D == DT->getDeducedType();

  return false;
}

static bool areTypesEquals(QualType TypeS, QualType TypeD,
                           bool IgnoreTypeAliases) {
  const QualType CTypeS = TypeS.getCanonicalType();
  const QualType CTypeD = TypeD.getCanonicalType();
  if (CTypeS != CTypeD)
    return false;

  return IgnoreTypeAliases || areTypesEquals(TypeS.getLocalUnqualifiedType(),
                                             TypeD.getLocalUnqualifiedType());
}

static bool areBinaryOperatorOperandsTypesEqual(const Expr *E,
                                                bool IgnoreTypeAliases) {
  if (!E)
    return false;
  const Expr *WithoutImplicitAndParen = E->IgnoreParenImpCasts();
  if (!WithoutImplicitAndParen)
    return false;
  if (const auto *B = dyn_cast<BinaryOperator>(WithoutImplicitAndParen)) {
    const QualType NonReferenceType =
        WithoutImplicitAndParen->getType().getNonReferenceType();
    if (!areTypesEquals(
            B->getLHS()->IgnoreImplicit()->getType().getNonReferenceType(),
            NonReferenceType, IgnoreTypeAliases))
      return true;
    if (!areTypesEquals(
            B->getRHS()->IgnoreImplicit()->getType().getNonReferenceType(),
            NonReferenceType, IgnoreTypeAliases))
      return true;
  }
  return false;
}

static const Decl *getSourceExprDecl(const Expr *SourceExpr) {
  const Expr *CleanSourceExpr = SourceExpr->IgnoreParenImpCasts();
  if (const auto *E = dyn_cast<DeclRefExpr>(CleanSourceExpr)) {
    return E->getDecl();
  }

  if (const auto *E = dyn_cast<CallExpr>(CleanSourceExpr)) {
    return E->getCalleeDecl();
  }

  if (const auto *E = dyn_cast<MemberExpr>(CleanSourceExpr)) {
    return E->getMemberDecl();
  }
  return nullptr;
}

RedundantCastingCheck::RedundantCastingCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)),
      IgnoreTypeAliases(Options.getLocalOrGlobal("IgnoreTypeAliases", false)) {}

void RedundantCastingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "IgnoreTypeAliases", IgnoreTypeAliases);
}

void RedundantCastingCheck::registerMatchers(MatchFinder *Finder) {

  auto SimpleType = qualType(hasCanonicalType(
      qualType(anyOf(builtinType(), references(builtinType()),
                     references(pointsTo(qualType())), pointsTo(qualType())))));

  auto BitfieldMemberExpr = memberExpr(member(fieldDecl(isBitField())));

  Finder->addMatcher(
      explicitCastExpr(
          unless(hasCastKind(CK_ConstructorConversion)),
          unless(hasCastKind(CK_UserDefinedConversion)),
          unless(cxxFunctionalCastExpr(hasDestinationType(unless(SimpleType)))),

          hasDestinationType(qualType().bind("type2")),
          hasSourceExpression(anyOf(
              expr(unless(initListExpr()), unless(BitfieldMemberExpr),
                   hasType(qualType().bind("type1")))
                  .bind("source"),
              initListExpr(unless(hasInit(1, expr())),
                           hasInit(0, expr(unless(BitfieldMemberExpr),
                                           hasType(qualType().bind("type1")))
                                          .bind("source"))))))
          .bind("cast"),
      this);
}

void RedundantCastingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SourceExpr = Result.Nodes.getNodeAs<Expr>("source");
  auto TypeD = *Result.Nodes.getNodeAs<QualType>("type2");

  if (SourceExpr->getValueKind() == VK_LValue &&
      TypeD.getCanonicalType()->isRValueReferenceType())
    return;

  const auto TypeS =
      Result.Nodes.getNodeAs<QualType>("type1")->getNonReferenceType();
  TypeD = TypeD.getNonReferenceType();

  if (!areTypesEquals(TypeS, TypeD, IgnoreTypeAliases))
    return;

  if (areBinaryOperatorOperandsTypesEqual(SourceExpr, IgnoreTypeAliases))
    return;

  const auto *CastExpr = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");
  if (IgnoreMacros &&
      (CastExpr->getBeginLoc().isMacroID() ||
       CastExpr->getEndLoc().isMacroID() || CastExpr->getExprLoc().isMacroID()))
    return;

  {
    auto Diag = diag(CastExpr->getExprLoc(),
                     "redundant explicit casting to the same type %0 as the "
                     "sub-expression, remove this casting");
    Diag << TypeD;

    const SourceManager &SM = *Result.SourceManager;
    const SourceLocation SourceExprBegin =
        SM.getExpansionLoc(SourceExpr->getBeginLoc());
    const SourceLocation SourceExprEnd =
        SM.getExpansionLoc(SourceExpr->getEndLoc());

    if (SourceExprBegin != CastExpr->getBeginLoc())
      Diag << FixItHint::CreateRemoval(SourceRange(
          CastExpr->getBeginLoc(), SourceExprBegin.getLocWithOffset(-1)));

    const SourceLocation NextToken = Lexer::getLocForEndOfToken(
        SourceExprEnd, 0U, SM, Result.Context->getLangOpts());

    if (SourceExprEnd != CastExpr->getEndLoc()) {
      Diag << FixItHint::CreateRemoval(
          SourceRange(NextToken, CastExpr->getEndLoc()));
    }

    if (utils::fixit::needParens(*SourceExpr)) {

      Diag << FixItHint::CreateInsertion(SourceExprBegin, "(")
           << FixItHint::CreateInsertion(NextToken, ")");
    }
  }

  const auto *SourceExprDecl = getSourceExprDecl(SourceExpr);
  if (!SourceExprDecl)
    return;

  if (const auto *D = dyn_cast<CXXConstructorDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from the invocation of this constructor",
         DiagnosticIDs::Note);
    return;
  }

  if (const auto *D = dyn_cast<FunctionDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from the invocation of this "
         "%select{function|method}0",
         DiagnosticIDs::Note)
        << isa<CXXMethodDecl>(D) << D->getReturnTypeSourceRange();
    return;
  }

  if (const auto *D = dyn_cast<FieldDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from referencing this member",
         DiagnosticIDs::Note)
        << SourceRange(D->getTypeSpecStartLoc(), D->getTypeSpecEndLoc());
    return;
  }

  if (const auto *D = dyn_cast<ParmVarDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from referencing this parameter",
         DiagnosticIDs::Note)
        << SourceRange(D->getTypeSpecStartLoc(), D->getTypeSpecEndLoc());
    return;
  }

  if (const auto *D = dyn_cast<VarDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from referencing this variable",
         DiagnosticIDs::Note)
        << SourceRange(D->getTypeSpecStartLoc(), D->getTypeSpecEndLoc());
    return;
  }

  if (const auto *D = dyn_cast<EnumConstantDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from referencing this enum constant",
         DiagnosticIDs::Note);
    return;
  }

  if (const auto *D = dyn_cast<BindingDecl>(SourceExprDecl)) {
    diag(D->getLocation(),
         "source type originates from referencing this bound variable",
         DiagnosticIDs::Note);
    return;
  }
}

} // namespace clang::tidy::readability
