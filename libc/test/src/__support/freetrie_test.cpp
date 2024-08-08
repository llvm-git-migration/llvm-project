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
  FreeTrie *trie = nullptr;
  FreeTrie *&empty_found = FreeTrie::find(trie, 123, {0, 1024});
  EXPECT_EQ(&empty_found, &trie);

  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  FreeTrie::push(trie, block1);

  FreeTrie *&root_found = FreeTrie::find(trie, block1->inner_size(), {0, 1024});
  EXPECT_EQ(&root_found, &trie);
}

} // namespace LIBC_NAMESPACE_DECL
