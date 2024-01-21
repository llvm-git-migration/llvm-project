//===-- SwapBinaryOperandsTests.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(SwapBinaryOperands);

TEST_F(SwapBinaryOperandsTest, Test) {
  Context = Function;
  EXPECT_EQ(apply("^p == nullptr"), "nullptr == p");
  EXPECT_EQ(apply("^p == nullptr"), "nullptr == p");
  EXPECT_EQ(apply("^x >= 5"), "5 <= x");
}

} // namespace
} // namespace clangd
} // namespace clang
