//===-- Unittests for a freelist --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freelist.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFreeList, PushPop) {
  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  cpp::byte mem2[1024];
  maybeBlock = Block<>::init(mem2);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block2 = *maybeBlock;

  FreeList *list = nullptr;
  FreeList::push(list, block1);
  ASSERT_NE(list, static_cast<FreeList *>(nullptr));
  EXPECT_EQ(list->block(), block1);

  FreeList::push(list, block2);
  EXPECT_EQ(list->block(), block1);

  FreeList::pop(list);
  ASSERT_NE(list, static_cast<FreeList *>(nullptr));
  EXPECT_EQ(list->block(), block2);

  FreeList::pop(list);
  ASSERT_EQ(list, static_cast<FreeList *>(nullptr));
}

} // namespace LIBC_NAMESPACE_DECL
