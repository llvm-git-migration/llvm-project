//===- llvm/unittest/Support/SipHashTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SipHash.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SipHashTest, PointerAuthSipHash) {
  // Test some basic cases.
  EXPECT_EQ(0xE793, getPointerAuthStableSipHash(""));
  EXPECT_EQ(0xF468, getPointerAuthStableSipHash("strlen"));
  EXPECT_EQ(0x2D15, getPointerAuthStableSipHash("_ZN1 ind; f"));

  // Test some known strings that are already enshrined in the ABI.
  EXPECT_EQ(0x6AE1, getPointerAuthStableSipHash("isa"));
  EXPECT_EQ(0xB5AB, getPointerAuthStableSipHash("objc_class:superclass"));
  EXPECT_EQ(0xC0BB, getPointerAuthStableSipHash("block_descriptor"));
  EXPECT_EQ(0xC310, getPointerAuthStableSipHash("method_list_t"));

  // Test limit cases where we differ from naive truncations from 64-bit hashes.
  EXPECT_EQ(1,      getPointerAuthStableSipHash("_Zptrkvttf"));
  EXPECT_EQ(0xFFFF, getPointerAuthStableSipHash("_Zaflhllod"));
}

} // end anonymous namespace
