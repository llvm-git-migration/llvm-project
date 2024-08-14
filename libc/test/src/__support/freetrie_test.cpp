//===-- Unittests for a freetrie --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freetrie.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFreeTrie, PushPop) {
  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  cpp::byte mem2[1024];
  maybeBlock = Block<>::init(mem2);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block2 = *maybeBlock;

  FreeTrie *trie = nullptr;
  FreeTrie::push(trie, block1);
  ASSERT_NE(trie, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(trie->block(), block1);

  FreeTrie::push(trie, block2);
  EXPECT_EQ(trie->block(), block1);

  FreeTrie::pop(trie);
  ASSERT_NE(trie, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(trie->block(), block2);

  FreeTrie::pop(trie);
  ASSERT_EQ(trie, static_cast<FreeTrie *>(nullptr));
}

TEST(LlvmLibcFreeTrie, Find) {
  size_t WIDTH = 1024;

  FreeTrie *trie = nullptr;
  FreeTrie *&empty_found = FreeTrie::find(trie, 123, {0, WIDTH});
  EXPECT_EQ(&empty_found, &trie);

  cpp::byte mem[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  FreeTrie::push(trie, block1);

  FreeTrie *&root_found =
      FreeTrie::find(trie, block1->inner_size(), {0, WIDTH});
  EXPECT_EQ(&root_found, &trie);

  FreeTrie *&less_found = FreeTrie::find(trie, WIDTH / 2, {0, 1024});
  EXPECT_NE(&less_found, &trie);
  EXPECT_EQ(less_found, static_cast<FreeTrie *>(nullptr));

  FreeTrie *&greater_found = FreeTrie::find(trie, WIDTH / 2 + 1, {0, 1024});
  EXPECT_NE(&greater_found, &trie);
  EXPECT_NE(&greater_found, &less_found);
  EXPECT_EQ(greater_found, static_cast<FreeTrie *>(nullptr));
}

TEST(LlvmLibcFreeTrie, RootPopWithChild) {
  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  cpp::byte mem2[1024];
  maybeBlock = Block<>::init(mem2);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block2 = *maybeBlock;

  cpp::byte mem3[2048];
  maybeBlock = Block<>::init(mem3);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block3 = *maybeBlock;

  cpp::byte mem4[2047];
  maybeBlock = Block<>::init(mem4);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block4 = *maybeBlock;

  FreeTrie *trie = nullptr;
  FreeTrie::push(trie, block1);
  FreeTrie::push(trie, block2);

  FreeTrie *&child3 = trie->find(trie, block3->inner_size(), {0, 4096});
  FreeTrie::push(child3, block3);

  ASSERT_NE(child3, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(child3->block(), block3);

  FreeTrie *&child4 = trie->find(trie, block4->inner_size(), {0, 4096});
  FreeTrie::push(child4, block4);
  ASSERT_NE(child4, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(child4->block(), block4);

  // Expected Trie:
  // block1 -> block2
  //   lower:
  //     block3
  //       upper:
  //         block4

  FreeTrie::pop(trie);
  FreeTrie *&new_child4 = trie->find(trie, block4->inner_size(), {0, 4096});
  // Expected Trie:
  // block2
  //   lower:
  //     block3
  //       upper:
  //         block4
  EXPECT_EQ(new_child4, child4);

  FreeTrie::pop(trie);

  // Expected Trie:
  // block4
  //   lower:
  //     block3
  EXPECT_EQ(trie, child4);
  FreeTrie *&new_child3 = trie->find(trie, block3->inner_size(), {0, 4096});
  EXPECT_EQ(new_child3, child3);
}

TEST(LlvmLibcFreeTrie, SizeRange) {
  FreeTrie::SizeRange range(123, 1024);
  EXPECT_EQ(range.min, size_t{123});
  EXPECT_EQ(range.width, size_t{1024});

  EXPECT_TRUE(range.contains(123));
  EXPECT_TRUE(range.contains(123 + 1024 - 1));
  EXPECT_FALSE(range.contains(123 - 1));
  EXPECT_FALSE(range.contains(123 + 1024 + 1));

  FreeTrie::SizeRange lower = range.lower();
  EXPECT_EQ(lower.min, size_t{123});
  EXPECT_EQ(lower.width, size_t{1024 / 2});

  FreeTrie::SizeRange upper = range.upper();
  EXPECT_EQ(upper.min, size_t{123 + 1024 / 2});
  EXPECT_EQ(upper.width, size_t{1024 / 2});
}

} // namespace LIBC_NAMESPACE_DECL
