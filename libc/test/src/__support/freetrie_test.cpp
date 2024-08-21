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

template <size_t size> struct BlockMem {
  BlockMem() {
    optional<Block<> *> maybeBlock = Block<>::init(mem);
    LIBC_ASSERT(maybeBlock.has_value() && "could not create test block");
    block = *maybeBlock;
  }
  __attribute__((aligned(alignof(Block<>)))) cpp::byte mem[size];
  Block<> *block;
};

TEST(LlvmLibcFreeTrie, PushPop) {
  BlockMem<1024> block1_mem;
  Block<> *block1 = block1_mem.block;
  BlockMem<1024> block2_mem;
  Block<> *block2 = block2_mem.block;
  BlockMem<1024> block3_mem;
  Block<> *block3 = block3_mem.block;

  FreeTrie *trie = nullptr;
  FreeTrie::push(trie, block1);
  ASSERT_NE(trie, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(trie->block(), block1);

  // Pushing blocks doesn't change the next block.
  FreeTrie::push(trie, block2);
  EXPECT_EQ(trie->block(), block1);
  FreeTrie::push(trie, block3);
  EXPECT_EQ(trie->block(), block1);

  // Blocks are popped in FIFO order
  FreeTrie::pop(trie);
  ASSERT_NE(trie, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(trie->block(), block2);
  FreeTrie::pop(trie);
  ASSERT_NE(trie, static_cast<FreeTrie *>(nullptr));
  EXPECT_EQ(trie->block(), block3);

  // Popping the last block clears the list.
  FreeTrie::pop(trie);
  ASSERT_EQ(trie, static_cast<FreeTrie *>(nullptr));
}

TEST(LlvmLibcFreeTrie, Find) {
  // Finding in an empty trie returns the trie itself.
  FreeTrie *trie = nullptr;
  FreeTrie *&empty_found = FreeTrie::find(trie, 123, {0, 1024});
  EXPECT_EQ(&empty_found, &trie);

  BlockMem<768> block_mem;
  Block<> *block = block_mem.block;
  FreeTrie::push(trie, block);

  // Finding the root by its exact size.
  FreeTrie *&root_found = FreeTrie::find(trie, block->inner_size(), {0, 1024});
  EXPECT_EQ(&root_found, &trie);

  // Sizes in the lower half of the range return one child.
  FreeTrie *&lower_found = FreeTrie::find(trie, 1024 / 2, {0, 1024});
  EXPECT_NE(&lower_found, &trie);
  EXPECT_EQ(lower_found, static_cast<FreeTrie *>(nullptr));

  FreeTrie *&lower2_found = FreeTrie::find(trie, 0, {0, 1024});
  EXPECT_EQ(&lower2_found, &lower_found);

  // Sizes in the upper half of the range return the other child.
  FreeTrie *&upper_found = FreeTrie::find(trie, 1024 / 2 + 1, {0, 1024});
  EXPECT_NE(&upper_found, &trie);
  EXPECT_NE(&upper_found, &lower_found);
  EXPECT_EQ(upper_found, static_cast<FreeTrie *>(nullptr));

  FreeTrie *&upper2_found = FreeTrie::find(trie, 1024 - 1, {0, 1024});
  EXPECT_EQ(&upper2_found, &upper_found);
}

TEST(LlvmLibcFreeTrie, PopPreservesChildren) {
  FreeTrie::SizeRange range{0, 4096};

  // Build the following trie:
  // 1 -> 2
  //   lower:
  //     3
  //       lower:
  //         4
  BlockMem<1024> block1_mem;
  Block<> *block1 = block1_mem.block;
  BlockMem<1024> block2_mem;
  Block<> *block2 = block2_mem.block;
  BlockMem<4096 / 2> block3_mem;
  Block<> *block3 = block3_mem.block;
  BlockMem<4096 / 2 - 1> block4_mem;
  Block<> *block4 = block4_mem.block;

  FreeTrie *trie = nullptr;
  FreeTrie::push(trie, block1);
  FreeTrie::push(trie, block2);
  FreeTrie *&child3 = FreeTrie::find(trie, block3->inner_size(), range);
  FreeTrie::push(child3, block3);
  FreeTrie *&child4 = FreeTrie::find(trie, block4->inner_size(), range);
  FreeTrie::push(child4, block4);

  // Popping an element from the root preserves the child links.
  FreeTrie::pop(trie);
  FreeTrie *&new_child4 = FreeTrie::find(trie, block4->inner_size(), range);
  EXPECT_EQ(new_child4, child4);

  // Popping the last element from the root moves a leaf (block4) to the root
  // and sets its children.
  FreeTrie::pop(trie);
  EXPECT_EQ(trie, child4);
  FreeTrie *&new_child3 = FreeTrie::find(trie, block3->inner_size(), range);
  EXPECT_EQ(new_child3, child3);
}

TEST(LlvmLibcFreeTrie, FindBestFitRoot) {
  FreeTrie::SizeRange range{0, 4096};

  FreeTrie *trie = nullptr;
  EXPECT_EQ(FreeTrie::find_best_fit(trie, 123, range),
            static_cast<FreeTrie **>(nullptr));

  BlockMem<1024> block_mem;
  Block<> *block = block_mem.block;
  FreeTrie::push(trie, block);

  EXPECT_EQ(FreeTrie::find_best_fit(trie, 0, range), &trie);
  EXPECT_EQ(FreeTrie::find_best_fit(trie, block->inner_size() - 1, range),
            &trie);
  EXPECT_EQ(FreeTrie::find_best_fit(trie, block->inner_size(), range), &trie);
  EXPECT_EQ(FreeTrie::find_best_fit(trie, block->inner_size() + 1, range),
            static_cast<FreeTrie **>(nullptr));
  EXPECT_EQ(FreeTrie::find_best_fit(trie, range.width - 1, range),
            static_cast<FreeTrie **>(nullptr));
}

TEST(LlvmLibcFreeTrie, FindBestFitLowerOnly) {
  FreeTrie::SizeRange range{0, 4096};

  FreeTrie *trie = nullptr;
  BlockMem<1024> root_mem;
  FreeTrie::push(trie, root_mem.block);
  BlockMem<1024 - 1> lower_mem;
  FreeTrie *&lower =
      FreeTrie::find(trie, lower_mem.block->inner_size(), range);
  FreeTrie::push(lower, lower_mem.block);

  EXPECT_EQ(FreeTrie::find_best_fit(trie, 0, range), &lower);
}

} // namespace LIBC_NAMESPACE_DECL
