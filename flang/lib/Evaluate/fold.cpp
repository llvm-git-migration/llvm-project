//===-- lib/Evaluate/fold.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/fold.h"
#include "fold-implementation.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/initial-image.h"
#include "flang/Evaluate/tools.h"

namespace Fortran::evaluate {

characteristics::TypeAndShape Fold(
    FoldingContext &context, characteristics::TypeAndShape &&x) {
  x.Rewrite(context);
  return std::move(x);
}

std::optional<Constant<SubscriptInteger>> GetConstantSubscript(
    FoldingContext &context, Subscript &ss, const NamedEntity &base, int dim) {
  ss = FoldOperation(context, std::move(ss));
  return common::visit(
      common::visitors{
          [](IndirectSubscriptIntegerExpr &expr)
              -> std::optional<Constant<SubscriptInteger>> {
            if (const auto *constant{
                    UnwrapConstantValue<SubscriptInteger>(expr.value())}) {
              return *constant;
            } else {
              return std::nullopt;
            }
          },
          [&](Triplet &triplet) -> std::optional<Constant<SubscriptInteger>> {
            auto lower{triplet.lower()}, upper{triplet.upper()};
            std::optional<ConstantSubscript> stride{ToInt64(triplet.stride())};
            if (!lower) {
              lower = GetLBOUND(context, base, dim);
            }
            if (!upper) {
              if (auto lb{GetLBOUND(context, base, dim)}) {
                upper = ComputeUpperBound(
                    context, std::move(*lb), GetExtent(context, base, dim));
              }
            }
            auto lbi{ToInt64(lower)}, ubi{ToInt64(upper)};
            if (lbi && ubi && stride && *stride != 0) {
              std::vector<SubscriptInteger::Scalar> values;
              while ((*stride > 0 && *lbi <= *ubi) ||
                  (*stride < 0 && *lbi >= *ubi)) {
                values.emplace_back(*lbi);
                *lbi += *stride;
              }
              return Constant<SubscriptInteger>{std::move(values),
                  ConstantSubscripts{
                      static_cast<ConstantSubscript>(values.size())}};
            } else {
              return std::nullopt;
            }
          },
      },
      ss.u);
}

Expr<SomeDerived> FoldOperation(
    FoldingContext &context, StructureConstructor &&structure) {
  StructureConstructor ctor{structure.derivedTypeSpec()};
  bool isConstant{true};
  auto restorer{context.WithPDTInstance(structure.derivedTypeSpec())};
  for (auto &&[symbol, value] : std::move(structure)) {
    auto expr{Fold(context, std::move(value.value()))};
    if (IsPointer(symbol)) {
      if (IsNullPointer(expr)) {
        // Handle x%c when x designates a named constant of derived
        // type and %c is NULL() in that constant.
        expr = Expr<SomeType>{NullPointer{}};
      } else if (IsProcedure(symbol)) {
        isConstant &= IsInitialProcedureTarget(expr);
      } else {
        isConstant &= IsInitialDataTarget(expr);
      }
    } else if (IsAllocatable(symbol)) {
      // F2023: 10.1.12 (3)(a)
      // If comp-spec is not null() for the allocatable component the
      // structure constructor is not a constant expression.
      isConstant &= IsNullPointer(expr);
    } else {
      isConstant &= IsActuallyConstant(expr) || IsNullPointer(expr);
      if (auto valueShape{GetConstantExtents(context, expr)}) {
        if (auto componentShape{GetConstantExtents(context, symbol)}) {
          if (GetRank(*componentShape) > 0 && GetRank(*valueShape) == 0) {
            expr = ScalarConstantExpander{std::move(*componentShape)}.Expand(
                std::move(expr));
            isConstant &= expr.Rank() > 0;
          } else {
            isConstant &= *valueShape == *componentShape;
          }
          if (*valueShape == *componentShape) {
            if (auto lbounds{AsConstantExtents(
                    context, GetLBOUNDs(context, NamedEntity{symbol}))}) {
              expr =
                  ArrayConstantBoundChanger{std::move(*lbounds)}.ChangeLbounds(
                      std::move(expr));
            }
          }
        }
      }
    }
    ctor.Add(symbol, std::move(expr));
  }
  if (isConstant) {
    return Expr<SomeDerived>{Constant<SomeDerived>{std::move(ctor)}};
  } else {
    return Expr<SomeDerived>{std::move(ctor)};
  }
}

Component FoldOperation(FoldingContext &context, Component &&component) {
  return {FoldOperation(context, std::move(component.base())),
      component.GetLastSymbol()};
}

NamedEntity FoldOperation(FoldingContext &context, NamedEntity &&x) {
  if (Component * c{x.UnwrapComponent()}) {
    return NamedEntity{FoldOperation(context, std::move(*c))};
  } else {
    return std::move(x);
  }
}

Triplet FoldOperation(FoldingContext &context, Triplet &&triplet) {
  MaybeExtentExpr lower{triplet.lower()};
  MaybeExtentExpr upper{triplet.upper()};
  return {Fold(context, std::move(lower)), Fold(context, std::move(upper)),
      Fold(context, triplet.stride())};
}

Subscript FoldOperation(FoldingContext &context, Subscript &&subscript) {
  return common::visit(
      common::visitors{
          [&](IndirectSubscriptIntegerExpr &&expr) {
            expr.value() = Fold(context, std::move(expr.value()));
            return Subscript(std::move(expr));
          },
          [&](Triplet &&triplet) {
            return Subscript(FoldOperation(context, std::move(triplet)));
          },
      },
      std::move(subscript.u));
}

ArrayRef FoldOperation(FoldingContext &context, ArrayRef &&arrayRef) {
  NamedEntity base{FoldOperation(context, std::move(arrayRef.base()))};
  for (Subscript &subscript : arrayRef.subscript()) {
    subscript = FoldOperation(context, std::move(subscript));
  }
  return ArrayRef{std::move(base), std::move(arrayRef.subscript())};
}

CoarrayRef FoldOperation(FoldingContext &context, CoarrayRef &&coarrayRef) {
  std::vector<Subscript> subscript;
  for (Subscript x : coarrayRef.subscript()) {
    subscript.emplace_back(FoldOperation(context, std::move(x)));
  }
  std::vector<Expr<SubscriptInteger>> cosubscript;
  for (Expr<SubscriptInteger> x : coarrayRef.cosubscript()) {
    cosubscript.emplace_back(Fold(context, std::move(x)));
  }
  CoarrayRef folded{std::move(coarrayRef.base()), std::move(subscript),
      std::move(cosubscript)};
  if (std::optional<Expr<SomeInteger>> stat{coarrayRef.stat()}) {
    folded.set_stat(Fold(context, std::move(*stat)));
  }
  if (std::optional<Expr<SomeInteger>> team{coarrayRef.team()}) {
    folded.set_team(
        Fold(context, std::move(*team)), coarrayRef.teamIsTeamNumber());
  }
  return folded;
}

DataRef FoldOperation(FoldingContext &context, DataRef &&dataRef) {
  return common::visit(common::visitors{
                           [&](SymbolRef symbol) { return DataRef{*symbol}; },
                           [&](auto &&x) {
                             return DataRef{
                                 FoldOperation(context, std::move(x))};
                           },
                       },
      std::move(dataRef.u));
}

Substring FoldOperation(FoldingContext &context, Substring &&substring) {
  auto lower{Fold(context, substring.lower())};
  auto upper{Fold(context, substring.upper())};
  if (const DataRef * dataRef{substring.GetParentIf<DataRef>()}) {
    return Substring{FoldOperation(context, DataRef{*dataRef}),
        std::move(lower), std::move(upper)};
  } else {
    auto p{*substring.GetParentIf<StaticDataObject::Pointer>()};
    return Substring{std::move(p), std::move(lower), std::move(upper)};
  }
}

ComplexPart FoldOperation(FoldingContext &context, ComplexPart &&complexPart) {
  DataRef complex{complexPart.complex()};
  return ComplexPart{
      FoldOperation(context, std::move(complex)), complexPart.part()};
}

std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &arg, std::int64_t defaultValue) {
  return arg ? ToInt64(*arg) : defaultValue;
}

Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&iDo) {
  if (std::optional<ConstantSubscript> value{context.GetImpliedDo(iDo.name)}) {
    return Expr<ImpliedDoIndex::Result>{*value};
  } else {
    return Expr<ImpliedDoIndex::Result>{std::move(iDo)};
  }
}

// TRANSFER (F'2018 16.9.193)
std::optional<Expr<SomeType>> FoldTransfer(
    FoldingContext &context, const ActualArguments &arguments) {
  CHECK(arguments.size() == 2 || arguments.size() == 3);
  const auto *source{UnwrapExpr<Expr<SomeType>>(arguments[0])};
  // There's a distinction between bytes in memory and bytes
  // in the Constant<> representation for REAL(10).
  std::optional<std::size_t> sourceConstantElementBytes;
  std::optional<std::size_t> sourceMemoryTotalBytes;
  std::optional<std::size_t> sourceConstantTotalBytes;
  if (source) {
    if (auto sourceTypeAndShape{
            characteristics::TypeAndShape::Characterize(*source, context)}) {
      std::optional<std::size_t> sourceMemoryElementBytes;
      if (auto sourceMemoryElementBytesExpr{
              sourceTypeAndShape->type().MeasureSizeInBytes(
                  context, /*aligned=*/true)}) {
        sourceMemoryElementBytes =
            ToInt64(Fold(context, std::move(*sourceMemoryElementBytesExpr)));
      }
      if (auto sourceConstantElementBytesExpr{
              sourceTypeAndShape->type().MeasureSizeInBytes(
                  context, /*aligned=*/false)}) {
        sourceConstantElementBytes =
            ToInt64(Fold(context, std::move(*sourceConstantElementBytesExpr)));
      }
      if (auto sourceElements{ToInt64(
              Fold(context, GetSize(Shape{sourceTypeAndShape->shape()})))}) {
        if (sourceMemoryElementBytes) {
          sourceMemoryTotalBytes = *sourceElements * *sourceMemoryElementBytes;
        }
        if (sourceConstantElementBytes) {
          sourceConstantTotalBytes =
              *sourceElements * *sourceConstantElementBytes;
          if (*sourceConstantElementBytes > 0 &&
              *sourceConstantTotalBytes / *sourceConstantElementBytes !=
                  static_cast<std::size_t>(*sourceElements)) {
            sourceConstantTotalBytes.reset();
          }
        }
      }
    }
  }
  std::optional<DynamicType> moldType;
  std::optional<std::int64_t> moldLength;
  if (arguments[1]) { // MOLD=
    moldType = arguments[1]->GetType();
    if (moldType && moldType->category() == TypeCategory::Character) {
      if (const auto *chExpr{UnwrapExpr<Expr<SomeCharacter>>(arguments[1])}) {
        moldLength = ToInt64(Fold(context, chExpr->LEN()));
      }
    }
  }
  std::optional<ConstantSubscripts> extents;
  std::optional<std::int64_t> moldBytes;
  if (moldType) {
    if (auto moldBytesExpr{
            moldType->MeasureSizeInBytes(context, /*align=*/true)}) {
      moldBytes = ToInt64(Fold(context, std::move(*moldBytesExpr)));
    }
  }
  if (arguments.size() == 2 && arguments[1]) { // no SIZE=
    if (arguments[1]->Rank() == 0) { // scalar MOLD=
      extents = ConstantSubscripts{}; // empty extents (scalar result)
    } else if (moldBytes.value_or(0) > 0 && sourceMemoryTotalBytes) {
      extents = ConstantSubscripts{static_cast<ConstantSubscript>(
          (*sourceMemoryTotalBytes + *moldBytes - 1) / *moldBytes)};
    }
  } else if (arguments.size() > 2) { // SIZE= is present
    if (auto sizeValue{ToInt64(arguments[2])}) {
      extents = ConstantSubscripts{*sizeValue};
    }
  }
  if (sourceConstantElementBytes && sourceConstantTotalBytes &&
      IsActuallyConstant(*source) && moldType && moldBytes && extents &&
      (moldLength || moldType->category() != TypeCategory::Character)) {
    auto resultElements{GetSize(*extents)};
    auto resultTotalBytes{*moldBytes * resultElements};
    // Don't fold intentional overflow cases from sneaky tests
    if (resultTotalBytes < 1000000 && resultElements < 1000000 &&
        (*moldBytes == 0 || resultTotalBytes / *moldBytes == resultElements)) {
      InitialImage image{*sourceConstantTotalBytes};
      auto status{image.Add(0, *sourceConstantTotalBytes, *source, context)};
      if (status == InitialImage::Ok) {
        return image.AsConstant(context, *moldType, moldLength, *extents,
            /*padWithZero=*/true,
            /*offset=*/0, /*elementBytes=*/*sourceConstantElementBytes);
      } else {
        // Can fail due to an allocatable or automatic component;
        // a warning will also have been produced.
        CHECK(status == InitialImage::NotAConstant);
      }
    }
  }
  return std::nullopt;
}

template class ExpressionBase<SomeDerived>;
template class ExpressionBase<SomeType>;

} // namespace Fortran::evaluate
