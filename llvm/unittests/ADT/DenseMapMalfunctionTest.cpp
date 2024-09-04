//===- llvm/unittest/ADT/DenseMapMalfunctionTest.cpp - DenseMap malfunction unit
// tests --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestIntAlloc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
template <> struct DenseMapInfo<IntAlloc> {
  static inline IntAlloc getEmptyKey() { return IntAlloc(0xFFFF); }
  static inline IntAlloc getTombstoneKey() { return IntAlloc(0xFFFF - 1); }
  static unsigned getHashValue(const IntAlloc &Val) {
    // magic value copied from llvm
    return Val.getValue() * 37U;
  }

  static bool isEqual(const IntAlloc &Lhs, const IntAlloc &Rhs) {
    return Lhs.getValue() == Rhs.getValue();
  }
};

template <> struct DenseMapInfo<IntAllocSpecial> {
  static inline IntAllocSpecial getEmptyKey() {
    return IntAllocSpecial(0xFFFF);
  }
  static inline IntAllocSpecial getTombstoneKey() {
    return IntAllocSpecial(0xFFFF - 1);
  }
  static unsigned getHashValue(const IntAlloc &Val) {
    // magic value copied from llvm
    return Val.getValue() * 37U;
  }

  static bool isEqual(const IntAllocSpecial &Lhs, const IntAllocSpecial &Rhs) {
    return Lhs.getValue() == Rhs.getValue();
  }
};

} // namespace llvm

namespace {

enum class EnumClass { Val };

// Test class
template <typename T> class DenseMapMalfunctionTest : public ::testing::Test {
protected:
  T Map;
};

// Register these types for testing.
// clang-format off
typedef ::testing::Types<DenseMap<uint32_t, uint32_t>,
                         DenseMap<uint32_t *, uint32_t *>,
                         DenseMap<EnumClass, uint32_t>,
                         SmallDenseMap<uint32_t, uint32_t>,
                         SmallDenseMap<uint32_t *, uint32_t *>,
                         SmallDenseMap<EnumClass, uint32_t>
                         > DenseMapMalfunctionTestTypes;
// clang-format on

TYPED_TEST_SUITE(DenseMapMalfunctionTest, DenseMapMalfunctionTestTypes, );

TYPED_TEST(DenseMapMalfunctionTest, SingleOperationTest1) {
  // insert
  this->Map.insert({1, 2});
  EXPECT_EQ(1, this->Map.size());
  EXPECT_EQ(2, this->Map.lookup(1));

  // update (noop)
  this->Map.insert({1, 3});
  EXPECT_EQ(1, this->Map.size());
  EXPECT_EQ(2, this->Map.lookup(1));

  // erase
  this->Map.erase({1});
  EXPECT_EQ(0, this->Map.size());
  EXPECT_FALSE(this->Map.contains(1));

  // clear
  this->Map.clear();
  EXPECT_EQ(0, this->Map.size());
  this->Map.insert({1, 3});
  EXPECT_EQ(1, this->Map.size());
  EXPECT_EQ(3, this->Map.lookup(1));
}

TYPED_TEST(DenseMapMalfunctionTest, SingleOperationTest2) {
  using MapType = typename TestFixture::T;

  // grow
  this->Map.grow(1024);

  // insert
  this->Map.insert({1, 2});
  EXPECT_EQ(1, this->Map.size());
  EXPECT_EQ(2, this->Map.lookup(1));

  // copy
  MapType CopiedDenseMap(this->Map);
  EXPECT_EQ(1, CopiedDenseMap.size());
  EXPECT_EQ(2, CopiedDenseMap.lookup(1));

  // move
  MapType MovedDenseMap(std::move(CopiedDenseMap));
  EXPECT_EQ(1, MovedDenseMap.size());
  EXPECT_EQ(2, MovedDenseMap.lookup(1));

  // shrink
  // add some elements before to exceed inplace buffer of SmallDenseMap
  this->Map.insert({2, 2});
  this->Map.insert({3, 2});
  this->Map.insert({4, 2});
  this->Map.insert({5, 2});
  this->Map.shrink_and_clear();
  EXPECT_EQ(0, this->Map.size());
}

TYPED_TEST(DenseMapMalfunctionTest, MassOperation) {
  uint32_t Count = 1024;
  for (uint32_t I = 0; I != Count; I++) {
    this->Map.insert({I, 2 * I});
    if (I % 2 == 0) {
      this->Map.erase(I);
    }
  }
  for (uint32_t I = 0; I != Count; I++) {
    if (I % 2 == 0) {
      EXPECT_FALSE(this->Map.contains(I));
    } else {
      EXPECT_EQ(2 * I, this->Map.at(I));
    }
  }
}

} // namespace
