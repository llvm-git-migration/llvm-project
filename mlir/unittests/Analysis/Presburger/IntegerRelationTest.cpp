//===- IntegerRelationTest.cpp - Tests for IntegerRelation class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "Parser.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(IntegerRelationTest, getDomainAndRangeSet) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, xr)[N] : (xr - x - 10 == 0, xr >= 0, N - xr >= 0)", 1);

  IntegerPolyhedron domainSet = rel.getDomainSet();

  IntegerPolyhedron expectedDomainSet =
      parseIntegerPolyhedron("(x)[N] : (x + 10 >= 0, N - x - 10 >= 0)");

  EXPECT_TRUE(domainSet.isEqual(expectedDomainSet));

  IntegerPolyhedron rangeSet = rel.getRangeSet();

  IntegerPolyhedron expectedRangeSet =
      parseIntegerPolyhedron("(x)[N] : (x >= 0, N - x >= 0)");

  EXPECT_TRUE(rangeSet.isEqual(expectedRangeSet));
}

TEST(IntegerRelationTest, inverse) {
  IntegerRelation rel =
      parseRelationFromSet("(x, y, z)[N, M] : (z - x - y == 0, x >= 0, N - x "
                           ">= 0, y >= 0, M - y >= 0)",
                           2);

  IntegerRelation inverseRel =
      parseRelationFromSet("(z, x, y)[N, M]  : (x >= 0, N - x >= 0, y >= 0, M "
                           "- y >= 0, x + y - z == 0)",
                           1);

  rel.inverse();

  EXPECT_TRUE(rel.isEqual(inverseRel));
}

TEST(IntegerRelationTest, intersectDomainAndRange) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
      ">= 0, x + y + z floordiv 7 == 0)",
      1);

  {
    IntegerPolyhedron poly =
        parseIntegerPolyhedron("(x)[N, M] : (x >= 0, M - x - 1 >= 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, x >= 0, M - x - 1 >= 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectDomain(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }

  {
    IntegerPolyhedron poly = parseIntegerPolyhedron(
        "(y, z)[N, M] : (y >= 0, M - y - 1 >= 0, y + z == 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, y >= 0, M - y - 1 >= 0, y + z == 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectRange(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }
}

TEST(IntegerRelationTest, applyDomainAndRange) {

  {
    IntegerRelation map1 = parseRelationFromSet(
        "(x, y, a, b)[N] : (a - x - N == 0, b - y + N == 0)", 2);
    IntegerRelation map2 =
        parseRelationFromSet("(x, y, a)[N] : (a - x - y == 0)", 2);

    map1.applyRange(map2);

    IntegerRelation map3 =
        parseRelationFromSet("(x, y, a)[N] : (a - x - y == 0)", 2);

    EXPECT_TRUE(map1.isEqual(map3));
  }

  {
    IntegerRelation map1 = parseRelationFromSet(
        "(x, y, a, b)[N] : (a - x + N == 0, b - y - N == 0)", 2);
    IntegerRelation map2 =
        parseRelationFromSet("(x, y, a, b)[N] : (a - N == 0, b - N == 0)", 2);

    IntegerRelation map3 =
        parseRelationFromSet("(x, y, a, b)[N] : (x - N == 0, y - N == 0)", 2);

    map1.applyDomain(map2);

    EXPECT_TRUE(map1.isEqual(map3));
  }
}

TEST(IntegerRelationTest, symbolicLexmin) {
  SymbolicLexOpt lexmin =
      parseRelationFromSet("(a, x)[b] : (x - a >= 0, x - b >= 0)", 1)
          .findSymbolicIntegerLexMin();

  PWMAFunction expectedLexmin = parsePWMAF({
      {"(a)[b] : (a - b >= 0)", "(a)[b] -> (a)"},     // a
      {"(a)[b] : (b - a - 1 >= 0)", "(a)[b] -> (b)"}, // b
  });
  EXPECT_TRUE(lexmin.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmin.lexopt.isEqual(expectedLexmin));
}

TEST(IntegerRelationTest, symbolicLexmax) {
  SymbolicLexOpt lexmax1 =
      parseRelationFromSet("(a, x)[b] : (a - x >= 0, b - x >= 0)", 1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax1 = parsePWMAF({
      {"(a)[b] : (a - b >= 0)", "(a)[b] -> (b)"},
      {"(a)[b] : (b - a - 1 >= 0)", "(a)[b] -> (a)"},
  });

  SymbolicLexOpt lexmax2 =
      parseRelationFromSet("(i, j)[N] : (i >= 0, j >= 0, N - i - j >= 0)", 1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax2 = parsePWMAF({
      {"(i)[N] : (i >= 0, N - i >= 0)", "(i)[N] -> (N - i)"},
  });

  SymbolicLexOpt lexmax3 =
      parseRelationFromSet("(x, y)[N] : (x >= 0, 2 * N - x >= 0, y >= 0, x - y "
                           "+ 2 * N >= 0, 4 * N - x - y >= 0)",
                           1)
          .findSymbolicIntegerLexMax();

  PWMAFunction expectedLexmax3 =
      parsePWMAF({{"(x)[N] : (x >= 0, 2 * N - x >= 0, x - N - 1 >= 0)",
                   "(x)[N] -> (4 * N - x)"},
                  {"(x)[N] : (x >= 0, 2 * N - x >= 0, -x + N >= 0)",
                   "(x)[N] -> (x + 2 * N)"}});

  EXPECT_TRUE(lexmax1.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax1.lexopt.isEqual(expectedLexmax1));
  EXPECT_TRUE(lexmax2.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax2.lexopt.isEqual(expectedLexmax2));
  EXPECT_TRUE(lexmax3.unboundedDomain.isIntegerEmpty());
  EXPECT_TRUE(lexmax3.lexopt.isEqual(expectedLexmax3));
}

TEST(IntegerRelationTest, convertVarKind) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(3, 3, 0);
  space.resetIds();

  // Attach identifiers.
  int identifiers[6] = {0, 1, 2, 3, 4, 5};
  space.getId(VarKind::SetDim, 0) = Identifier(&identifiers[0]);
  space.getId(VarKind::SetDim, 1) = Identifier(&identifiers[1]);
  space.getId(VarKind::SetDim, 2) = Identifier(&identifiers[2]);
  space.getId(VarKind::Symbol, 0) = Identifier(&identifiers[3]);
  space.getId(VarKind::Symbol, 1) = Identifier(&identifiers[4]);
  space.getId(VarKind::Symbol, 2) = Identifier(&identifiers[5]);

  // Cannot call parseIntegerRelation to test convertVarKind as
  // parseIntegerRelation uses convertVarKind.
  IntegerRelation rel = parseIntegerPolyhedron(
      // 0  1  2  3  4  5
      "(x, y, a)[U, V, W] : (x - U == 0, y + a - W == 0, U - V >= 0,"
      "y - a >= 0)");
  rel.setSpace(space);

  // Make a few kind conversions.
  rel.convertVarKind(VarKind::Symbol, 1, 2, VarKind::Domain, 0);
  rel.convertVarKind(VarKind::Range, 2, 3, VarKind::Domain, 0);
  rel.convertVarKind(VarKind::Range, 0, 2, VarKind::Symbol, 1);
  rel.convertVarKind(VarKind::Domain, 1, 2, VarKind::Range, 0);
  rel.convertVarKind(VarKind::Domain, 0, 1, VarKind::Range, 1);

  space = rel.getSpace();

  // Expected rel.
  IntegerRelation expectedRel = parseIntegerPolyhedron(
      "(V, a)[U, x, y, W] : (x - U == 0, y + a - W == 0, U - V >= 0,"
      "y - a >= 0)");
  expectedRel.setSpace(space);

  EXPECT_TRUE(rel.isEqual(expectedRel));

  EXPECT_EQ(space.getId(VarKind::SetDim, 0), Identifier(&identifiers[4]));
  EXPECT_EQ(space.getId(VarKind::SetDim, 1), Identifier(&identifiers[2]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 0), Identifier(&identifiers[3]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 1), Identifier(&identifiers[0]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 2), Identifier(&identifiers[1]));
  EXPECT_EQ(space.getId(VarKind::Symbol, 3), Identifier(&identifiers[5]));
}

TEST(IntegerRelationTest, convertVarKindToLocal) {
  // Convert all range variables to local variables
  IntegerRelation rel =
      parseRelationFromSet("(x, y, z) : (x >= 0, y >= 0, -z >= 0)", 1);
  rel.convertToLocal(VarKind::Range, 0, rel.getNumRangeVars());
  EXPECT_FALSE(rel.isEmptyByGCDTest());

  // Convert all domain variables to local variables
  IntegerRelation rel2 =
      parseRelationFromSet("(x, y, z) : (x >= 0, y >= 0, -z >= 0)", 2);
  rel2.convertToLocal(VarKind::Domain, 0, rel.getNumDomainVars());
  EXPECT_FALSE(rel2.isEmptyByGCDTest());

  // Convert a prefix of range variables to local variables
  IntegerRelation rel3 = parseRelationFromSet(
      "(x, y, u, v) : (x >= 0, y >= 0, -u >= 0, -v >= 0)", 2);
  rel3.convertToLocal(VarKind::Range, rel.getNumDomainVars(), 1);
  EXPECT_FALSE(rel3.isEmptyByGCDTest());

  // Convert a suffix of domain variables to local variables
  IntegerRelation rel4 = parseRelationFromSet(
      "(x, y, u, v) : (x >= 0, y >= 0, -u >= 0, -v >= 0)", 2);
  rel4.convertToLocal(VarKind::Domain, rel.getNumDomainVars() - 1,
                      rel.getNumDomainVars());
  EXPECT_FALSE(rel4.isEmptyByGCDTest());
}
