//===-- Unittests for a block of memory -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stddef.h>

#include "src/stdlib/block.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/span.h"
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

// Block types.
using LargeOffsetBlock = LIBC_NAMESPACE::Block<uint64_t>;
using SmallOffsetBlock = LIBC_NAMESPACE::Block<uint16_t>;

// For each of the block types above, we'd like to run the same tests since
// they should work independently of the parameter sizes. Rather than re-writing
// the same test for each case, let's instead create a custom test framework for
// each test case that invokes the actual testing function for each block type.
//
// It's organized this way because the ASSERT/EXPECT macros only work within a
// `Test` class due to those macros expanding to `test` methods.
#define TEST_FOR_EACH_BLOCK_TYPE(TestCase)                                     \
  class LlvmLibcBlockTest##TestCase : public LIBC_NAMESPACE::testing::Test {   \
  public:                                                                      \
    template <typename BlockType> void RunTest();                              \
  };                                                                           \
  TEST_F(LlvmLibcBlockTest##TestCase, TestCase) {                              \
    RunTest<LargeOffsetBlock>();                                               \
    RunTest<SmallOffsetBlock>();                                               \
  }                                                                            \
  template <typename BlockType> void LlvmLibcBlockTest##TestCase::RunTest()

using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::span;

TEST_FOR_EACH_BLOCK_TYPE(CanCreateSingleAlignedBlock) {
  constexpr size_t kN = 1024;
  alignas(BlockType::kAlignment) array<byte, kN> bytes;

  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  EXPECT_EQ(block->OuterSize(), kN);
  EXPECT_EQ(block->InnerSize(), kN - BlockType::kBlockOverhead);
  EXPECT_EQ(block->Prev(), static_cast<BlockType *>(nullptr));
  EXPECT_EQ(block->Next(), static_cast<BlockType *>(nullptr));
  EXPECT_FALSE(block->Used());
  EXPECT_TRUE(block->Last());
}

TEST_FOR_EACH_BLOCK_TYPE(CanCreateUnalignedSingleBlock) {
  constexpr size_t kN = 1024;

  // Force alignment, so we can un-force it below
  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  span<byte> aligned(bytes);

  auto result = BlockType::Init(aligned.subspan(1));
  EXPECT_TRUE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotCreateTooSmallBlock) {
  array<byte, 2> bytes;
  auto result = BlockType::Init(bytes);
  EXPECT_FALSE(result.has_value());
}

// This test specifically checks that we cannot allocate a block with a size
// larger than what can be held by the offset type, we don't need to test with
// multiple block types for this particular check, so we use the normal TEST
// macro and not the custom framework.
TEST(LlvmLibcBlockTest, CannotCreateTooLargeBlock) {
  using BlockType = LIBC_NAMESPACE::Block<uint8_t>;
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  EXPECT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  auto *block1 = *result;

  result = BlockType::Split(block1, kSplitN);
  ASSERT_TRUE(result.has_value());

  auto *block2 = *result;

  EXPECT_EQ(block1->InnerSize(), kSplitN);
  EXPECT_EQ(block1->OuterSize(), kSplitN + BlockType::kBlockOverhead);
  EXPECT_FALSE(block1->Last());

  EXPECT_EQ(block2->OuterSize(), kN - kSplitN - BlockType::kBlockOverhead);
  EXPECT_FALSE(block2->Used());
  EXPECT_TRUE(block2->Last());

  EXPECT_EQ(block1->Next(), block2);
  EXPECT_EQ(block2->Prev(), block1);
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitBlockUnaligned) {
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  // We should split at sizeof(BlockType) + kSplitN bytes. Then
  // we need to round that up to an alignof(BlockType) boundary.
  constexpr size_t kSplitN = 513;
  uintptr_t split_addr = reinterpret_cast<uintptr_t>(block1) + kSplitN;
  split_addr += alignof(BlockType) - (split_addr % alignof(BlockType));
  uintptr_t split_len = split_addr - (uintptr_t)&bytes;

  result = BlockType::Split(block1, kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_EQ(block1->InnerSize(), split_len);
  EXPECT_EQ(block1->OuterSize(), split_len + BlockType::kBlockOverhead);

  EXPECT_EQ(block2->OuterSize(), kN - block1->OuterSize());
  EXPECT_FALSE(block2->Used());

  EXPECT_EQ(block1->Next(), block2);
  EXPECT_EQ(block2->Prev(), block1);
}

TEST_FOR_EACH_BLOCK_TYPE(CanSplitMidBlock) {
  // Split once, then split the original block again to ensure that the
  // pointers get rewired properly.
  // I.e.
  // [[             BLOCK 1            ]]
  // block1->Split()
  // [[       BLOCK1       ]][[ BLOCK2 ]]
  // block1->Split()
  // [[ BLOCK1 ]][[ BLOCK3 ]][[ BLOCK2 ]]

  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block1, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  EXPECT_EQ(block1->Next(), block3);
  EXPECT_EQ(block3->Prev(), block1);
  EXPECT_EQ(block3->Next(), block2);
  EXPECT_EQ(block2->Prev(), block3);
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitTooSmallBlock) {
  constexpr size_t kN = 64;
  constexpr size_t kSplitN = kN + 1;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = BlockType::Split(block, kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitBlockWithoutHeaderSpace) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = kN - BlockType::kBlockOverhead - 1;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = BlockType::Split(block, kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitNull) {
  BlockType *block = nullptr;
  auto result = BlockType::Split(block, 1);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMakeBlockLargerInSplit) {
  // Ensure that we can't ask for more space than the block actually has...
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = BlockType::Split(block, block->InnerSize() + 1);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMakeSecondBlockLargerInSplit) {
  // Ensure that the second block in split is at least of the size of header.
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = BlockType::Split(block,
                            block->InnerSize() - BlockType::kBlockOverhead + 1);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanMakeZeroSizeFirstBlock) {
  // This block does support splitting with zero payload size.
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  result = BlockType::Split(block, 0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(block->InnerSize(), static_cast<size_t>(0));
}

TEST_FOR_EACH_BLOCK_TYPE(CanMakeZeroSizeSecondBlock) {
  // Likewise, the split block can be zero-width.
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result =
      BlockType::Split(block1, block1->InnerSize() - BlockType::kBlockOverhead);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_EQ(block2->InnerSize(), static_cast<size_t>(0));
}

TEST_FOR_EACH_BLOCK_TYPE(CanMarkBlockUsed) {
  constexpr size_t kN = 1024;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  block->MarkUsed();
  EXPECT_TRUE(block->Used());

  // Size should be unaffected.
  EXPECT_EQ(block->OuterSize(), kN);

  block->MarkFree();
  EXPECT_FALSE(block->Used());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotSplitUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  block->MarkUsed();
  result = BlockType::Split(block, kSplitN);
  ASSERT_FALSE(result.has_value());
}

TEST_FOR_EACH_BLOCK_TYPE(CanMergeWithNextBlock) {
  // Do the three way merge from "CanSplitMidBlock", and let's
  // merge block 3 and 2
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());

  result = BlockType::Split(block1, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  EXPECT_TRUE(BlockType::MergeNext(block3));

  EXPECT_EQ(block1->Next(), block3);
  EXPECT_EQ(block3->Prev(), block1);
  EXPECT_EQ(block1->InnerSize(), kSplit2);
  EXPECT_EQ(block3->OuterSize(), kN - block1->OuterSize());
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMergeWithFirstOrLastBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  // Do a split, just to check that the checks on Next/Prev are different...
  result = BlockType::Split(block1, kSplitN);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  EXPECT_FALSE(BlockType::MergeNext(block2));
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMergeNull) {
  BlockType *block = nullptr;
  EXPECT_FALSE(BlockType::MergeNext(block));
}

TEST_FOR_EACH_BLOCK_TYPE(CannotMergeUsedBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplitN = 512;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  // Do a split, just to check that the checks on Next/Prev are different...
  result = BlockType::Split(block, kSplitN);
  ASSERT_TRUE(result.has_value());

  block->MarkUsed();
  EXPECT_FALSE(BlockType::MergeNext(block));
}

TEST_FOR_EACH_BLOCK_TYPE(CanFreeSingleBlock) {
  constexpr size_t kN = 1024;
  alignas(BlockType::kAlignment) array<byte, kN> bytes;

  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block = *result;

  block->MarkUsed();
  BlockType::Free(block);
  EXPECT_FALSE(block->Used());
  EXPECT_EQ(block->OuterSize(), kN);
}

TEST_FOR_EACH_BLOCK_TYPE(CanFreeBlockWithoutMerging) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  block1->MarkUsed();
  block2->MarkUsed();
  block3->MarkUsed();

  BlockType::Free(block2);
  EXPECT_FALSE(block2->Used());
  EXPECT_NE(block2->Prev(), static_cast<BlockType *>(nullptr));
  EXPECT_FALSE(block2->Last());
}

TEST_FOR_EACH_BLOCK_TYPE(CanFreeBlockAndMergeWithPrev) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  block2->MarkUsed();
  block3->MarkUsed();

  BlockType::Free(block2);
  EXPECT_FALSE(block2->Used());
  EXPECT_EQ(block2->Prev(), static_cast<BlockType *>(nullptr));
  EXPECT_FALSE(block2->Last());
}

TEST_FOR_EACH_BLOCK_TYPE(CanFreeBlockAndMergeWithNext) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());

  block1->MarkUsed();
  block2->MarkUsed();

  BlockType::Free(block2);
  EXPECT_FALSE(block2->Used());
  EXPECT_NE(block2->Prev(), static_cast<BlockType *>(nullptr));
  EXPECT_TRUE(block2->Last());
}

TEST_FOR_EACH_BLOCK_TYPE(CanFreeUsedBlockAndMergeWithBoth) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());

  block2->MarkUsed();

  BlockType::Free(block2);
  EXPECT_FALSE(block2->Used());
  EXPECT_EQ(block2->Prev(), static_cast<BlockType *>(nullptr));
  EXPECT_TRUE(block2->Last());
}

TEST_FOR_EACH_BLOCK_TYPE(CanCheckValidBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 512;
  constexpr size_t kSplit2 = 256;

  alignas(BlockType::kAlignment) array<byte, kN> bytes;
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  EXPECT_TRUE(block1->IsValid());
  EXPECT_TRUE(block2->IsValid());
  EXPECT_TRUE(block3->IsValid());
}

TEST_FOR_EACH_BLOCK_TYPE(CanCheckInvalidBlock) {
  constexpr size_t kN = 1024;
  constexpr size_t kSplit1 = 128;
  constexpr size_t kSplit2 = 384;
  constexpr size_t kSplit3 = 256;

  array<byte, kN> bytes{};
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  result = BlockType::Split(block1, kSplit1);
  ASSERT_TRUE(result.has_value());
  BlockType *block2 = *result;

  result = BlockType::Split(block2, kSplit2);
  ASSERT_TRUE(result.has_value());
  BlockType *block3 = *result;

  result = BlockType::Split(block3, kSplit3);
  ASSERT_TRUE(result.has_value());

  // Corrupt a Block header.
  // This must not touch memory outside the original region, or the test may
  // (correctly) abort when run with address sanitizer.
  // To remain as agostic to the internals of `Block` as possible, the test
  // copies a smaller block's header to a larger block.
  EXPECT_TRUE(block1->IsValid());
  EXPECT_TRUE(block2->IsValid());
  EXPECT_TRUE(block3->IsValid());
  auto *src = reinterpret_cast<byte *>(block1);
  auto *dst = reinterpret_cast<byte *>(block2);
  LIBC_NAMESPACE::memcpy(dst, src, sizeof(BlockType));
  EXPECT_FALSE(block1->IsValid());
  EXPECT_FALSE(block2->IsValid());
  EXPECT_FALSE(block3->IsValid());
}

TEST_FOR_EACH_BLOCK_TYPE(CanGetBlockFromUsableSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes{};
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  BlockType *block1 = *result;

  void *ptr = block1->UsableSpace();
  BlockType *block2 = BlockType::FromUsableSpace(ptr);
  EXPECT_EQ(block1, block2);
}

TEST_FOR_EACH_BLOCK_TYPE(CanGetConstBlockFromUsableSpace) {
  constexpr size_t kN = 1024;

  array<byte, kN> bytes{};
  auto result = BlockType::Init(bytes);
  ASSERT_TRUE(result.has_value());
  const BlockType *block1 = *result;

  const void *ptr = block1->UsableSpace();
  const BlockType *block2 = BlockType::FromUsableSpace(ptr);
  EXPECT_EQ(block1, block2);
}
