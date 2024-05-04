//===- PolynomialAttributes.cpp - Polynomial dialect attrs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"

#include "mlir/Dialect/Polynomial/IR/Polynomial.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace polynomial {

void IntPolynomialAttr::print(AsmPrinter &p) const {
  p << '<' << getPolynomial() << '>';
}

void FloatPolynomialAttr::print(AsmPrinter &p) const {
  p << '<' << getPolynomial() << '>';
}

/// A callable that parses the coefficient using the appropriate method for the
/// given monomial type, and stores the parsed coefficient value on the
/// monomial.
template <typename CoefficientType>
using ParseCoefficientFn =
    std::function<OptionalParseResult(CoefficientType &)>;

/// Try to parse a monomial. If successful, populate the fields of the outparam
/// `monomial` with the results, and the `variable` outparam with the parsed
/// variable name. Sets shouldParseMore to true if the monomial is followed by
/// a '+'.
///
template <typename Monomial, typename CoefficientType>
ParseResult
parseMonomial(AsmParser &parser, Monomial &monomial, llvm::StringRef &variable,
              bool &isConstantTerm, bool &shouldParseMore,
              ParseCoefficientFn<CoefficientType> parseAndStoreCoefficient) {
  OptionalParseResult parsedCoeffResult =
      parseAndStoreCoefficient(monomial.getMutableCoefficient());

  isConstantTerm = false;
  shouldParseMore = false;

  // A + indicates it's a constant term with more to go, as in `1 + x`.
  if (succeeded(parser.parseOptionalPlus())) {
    // If no coefficient was parsed, and there's a +, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }
    monomial.setExponent(APInt(apintBitWidth, 0));
    isConstantTerm = true;
    shouldParseMore = true;
    return success();
  }

  // A monomial can be a trailing constant term, as in `x + 1`.
  if (failed(parser.parseOptionalKeyword(&variable))) {
    // If neither a coefficient nor a variable was found, then it's effectively
    // parsing an empty string.
    if (!parsedCoeffResult.has_value()) {
      return failure();
    }

    monomial.setExponent(APInt(apintBitWidth, 0));
    isConstantTerm = true;
    return success();
  }

  // Parse exponentiation symbol as `**`. We can't use caret because it's
  // reserved for basic block identifiers If no star is present, it's treated
  // as a polynomial with exponent 1.
  if (succeeded(parser.parseOptionalStar())) {
    // If there's one * there must be two.
    if (failed(parser.parseStar())) {
      return failure();
    }

    // If there's a **, then the integer exponent is required.
    APInt parsedExponent(apintBitWidth, 0);
    if (failed(parser.parseInteger(parsedExponent))) {
      parser.emitError(parser.getCurrentLocation(),
                       "found invalid integer exponent");
      return failure();
    }

    monomial.setExponent(parsedExponent);
  } else {
    monomial.setExponent(APInt(apintBitWidth, 1));
  }

  if (succeeded(parser.parseOptionalPlus())) {
    shouldParseMore = true;
  }
  return success();
}

template <typename PolynoimalAttrTy, typename Monomial, typename CoefficientTy>
LogicalResult parsePolynomialAttr(
    AsmParser &parser, llvm::SmallVector<Monomial> &monomials,
    llvm::StringSet<> &variables,
    ParseCoefficientFn<CoefficientTy> parseAndStoreCoefficient) {
  while (true) {
    Monomial parsedMonomial;
    llvm::StringRef parsedVariableRef;
    bool isConstantTerm;
    bool shouldParseMore;
    if (failed(parseMonomial<Monomial, CoefficientTy>(
            parser, parsedMonomial, parsedVariableRef, isConstantTerm,
            shouldParseMore, parseAndStoreCoefficient))) {
      parser.emitError(parser.getCurrentLocation(), "expected a monomial");
      return failure();
    }

    if (!isConstantTerm) {
      std::string parsedVariable = parsedVariableRef.str();
      variables.insert(parsedVariable);
    }
    monomials.push_back(parsedMonomial);

    if (shouldParseMore)
      continue;

    if (succeeded(parser.parseOptionalGreater())) {
      break;
    }
    parser.emitError(
        parser.getCurrentLocation(),
        "expected + and more monomials, or > to end polynomial attribute");
    return failure();
  }

  if (variables.size() > 1) {
    std::string vars = llvm::join(variables.keys(), ", ");
    parser.emitError(
        parser.getCurrentLocation(),
        "polynomials must have one indeterminate, but there were multiple: " +
            vars);
    return failure();
  }

  return success();
}

Attribute IntPolynomialAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<IntMonomial> monomials;
  llvm::StringSet<> variables;

  if (failed(parsePolynomialAttr<IntPolynomialAttr, IntMonomial, APInt>(
          parser, monomials, variables,
          [&](APInt &coeff) -> OptionalParseResult {
            return parser.parseOptionalInteger(coeff);
          }))) {
    return {};
  }

  auto result = IntPolynomial::fromMonomials(monomials);
  if (failed(result)) {
    parser.emitError(parser.getCurrentLocation())
        << "parsed polynomial must have unique exponents among monomials";
    return {};
  }
  return IntPolynomialAttr::get(parser.getContext(), result.value());
}

Attribute FloatPolynomialAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<FloatMonomial> monomials;
  llvm::StringSet<> variables;

  ParseCoefficientFn<APFloat> parseAndStoreCoefficient =
      [&](APFloat &coeff) -> OptionalParseResult {
    double coeffValue;
    ParseResult result = parser.parseFloat(coeffValue);
    if (succeeded(result)) {
      coeff = APFloat(coeffValue);
    }
    return OptionalParseResult(result);
  };

  if (failed(parsePolynomialAttr<FloatPolynomialAttr, FloatMonomial, APFloat>(
          parser, monomials, variables, parseAndStoreCoefficient))) {
    return {};
  }

  auto result = FloatPolynomial::fromMonomials(monomials);
  if (failed(result)) {
    parser.emitError(parser.getCurrentLocation())
        << "parsed polynomial must have unique exponents among monomials";
    return {};
  }
  return FloatPolynomialAttr::get(parser.getContext(), result.value());
}

void RingAttr::print(AsmPrinter &p) const {
  p << "#polynomial.ring<coefficientType=" << getCoefficientType()
    << ", coefficientModulus=" << getCoefficientModulus()
    << ", polynomialModulus=" << getPolynomialModulus() << '>';
}

Attribute RingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  if (failed(parser.parseKeyword("coefficientType")))
    return {};

  if (failed(parser.parseEqual()))
    return {};

  Type ty;
  if (failed(parser.parseType(ty)))
    return {};

  if (failed(parser.parseComma()))
    return {};

  IntegerAttr coefficientModulusAttr = nullptr;
  if (succeeded(parser.parseKeyword("coefficientModulus"))) {
    if (failed(parser.parseEqual()))
      return {};

    IntegerType iType = mlir::dyn_cast<IntegerType>(ty);
    if (!iType) {
      parser.emitError(parser.getCurrentLocation(),
                       "coefficientType must specify an integer type");
      return {};
    }
    APInt coefficientModulus(iType.getWidth(), 0);
    auto result = parser.parseInteger(coefficientModulus);
    if (failed(result)) {
      parser.emitError(parser.getCurrentLocation(),
                       "invalid coefficient modulus");
      return {};
    }
    coefficientModulusAttr = IntegerAttr::get(iType, coefficientModulus);

    if (failed(parser.parseComma()))
      return {};
  }

  IntPolynomialAttr polyAttr = nullptr;
  if (succeeded(parser.parseKeyword("polynomialModulus"))) {
    if (failed(parser.parseEqual()))
      return {};

    IntPolynomialAttr attr;
    if (failed(parser.parseAttribute<IntPolynomialAttr>(attr)))
      return {};
    polyAttr = attr;
  }

  IntPolynomial poly = polyAttr.getPolynomial();
  APInt root(coefficientModulusAttr.getValue().getBitWidth(), 0);
  IntegerAttr rootAttr = nullptr;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("primitiveRoot")) ||
        failed(parser.parseEqual()))
      return {};

    ParseResult result = parser.parseInteger(root);
    if (failed(result)) {
      parser.emitError(parser.getCurrentLocation(), "invalid primitiveRoot");
      return {};
    }
    rootAttr = IntegerAttr::get(coefficientModulusAttr.getType(), root);
  }

  if (failed(parser.parseGreater()))
    return {};

  return RingAttr::get(parser.getContext(), ty, coefficientModulusAttr,
                       polyAttr, rootAttr);
}

} // namespace polynomial
} // namespace mlir
