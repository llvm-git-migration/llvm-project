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

TEST(LlvmLibcFreeTrie, Construct) {
  FreeTrie trie;
  EXPECT_TRUE(trie.empty());
}

TEST(LlvmLibcFreeTrie, Push) {
  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block = *maybeBlock;

  FreeTrie trie;
  trie.push(block);
  ASSERT_FALSE(trie.empty());
  EXPECT_EQ(trie.front(), block);
}

} // namespace LIBC_NAMESPACE_DECL
