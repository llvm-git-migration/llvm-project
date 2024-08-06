//===-- Unittests for a freelist --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freelist2.h"
#include "src/__support/freelist2.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFreeList2, DefaultListIsEmpty) {
  FreeList2 list;
  EXPECT_TRUE(list.empty());
}

TEST(LlvmLibcFreeList2, PushMakesListNonEmpty) {
  cpp::byte mem[1024];
  optional<Block<>*> maybeBlock = Block<>::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<>* block = *maybeBlock;

  FreeList2 list;
  list.push(block);
  EXPECT_FALSE(list.empty());
}

TEST(LlvmLibcFreeList2, PushPopEmptiesList) {
  cpp::byte mem[1024];
  optional<Block<>*> maybeBlock = Block<>::init(mem);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<>* block = *maybeBlock;

  FreeList2 list;
  list.push(block);
  list.pop();
  EXPECT_TRUE(list.empty());
}

#if 0
TEST(LlvmLibcFreeList2, EmptyListHasNoMembers) {
  FreeList<SIZE> list(example_sizes);

  auto item = list.find_chunk(4);
  EXPECT_EQ(item.size(), static_cast<size_t>(0));
  item = list.find_chunk(128);
  EXPECT_EQ(item.size(), static_cast<size_t>(0));
}

TEST(LlvmLibcFreeList, CanRetrieveAddedMember) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t N = 512;

  byte data[N] = {byte(0)};

  bool ok = list.add_chunk(span<byte>(data, N));
  EXPECT_TRUE(ok);

  auto item = list.find_chunk(N);
  EXPECT_EQ(item.size(), N);
  EXPECT_EQ(item.data(), data);
}

TEST(LlvmLibcFreeList, CanRetrieveAddedMemberForSmallerSize) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t N = 512;

  byte data[N] = {byte(0)};

  ASSERT_TRUE(list.add_chunk(span<byte>(data, N)));
  auto item = list.find_chunk(N / 2);
  EXPECT_EQ(item.size(), N);
  EXPECT_EQ(item.data(), data);
}

TEST(LlvmLibcFreeList, CanRemoveItem) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t N = 512;

  byte data[N] = {byte(0)};

  ASSERT_TRUE(list.add_chunk(span<byte>(data, N)));
  EXPECT_TRUE(list.remove_chunk(span<byte>(data, N)));

  auto item = list.find_chunk(N);
  EXPECT_EQ(item.size(), static_cast<size_t>(0));
}

TEST(LlvmLibcFreeList, FindReturnsSmallestChunk) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t kN1 = 512;
  constexpr size_t kN2 = 1024;

  byte data1[kN1] = {byte(0)};
  byte data2[kN2] = {byte(0)};

  ASSERT_TRUE(list.add_chunk(span<byte>(data1, kN1)));
  ASSERT_TRUE(list.add_chunk(span<byte>(data2, kN2)));

  auto chunk = list.find_chunk(kN1 / 2);
  EXPECT_EQ(chunk.size(), kN1);
  EXPECT_EQ(chunk.data(), data1);

  chunk = list.find_chunk(kN1);
  EXPECT_EQ(chunk.size(), kN1);
  EXPECT_EQ(chunk.data(), data1);

  chunk = list.find_chunk(kN1 + 1);
  EXPECT_EQ(chunk.size(), kN2);
  EXPECT_EQ(chunk.data(), data2);
}

TEST(LlvmLibcFreeList, FindReturnsCorrectChunkInSameBucket) {
  // If we have two values in the same bucket, ensure that the allocation will
  // pick an appropriately sized one.
  FreeList<SIZE> list(example_sizes);
  constexpr size_t kN1 = 512;
  constexpr size_t kN2 = 257;

  byte data1[kN1] = {byte(0)};
  byte data2[kN2] = {byte(0)};

  // List should now be 257 -> 512 -> NULL
  ASSERT_TRUE(list.add_chunk(span<byte>(data1, kN1)));
  ASSERT_TRUE(list.add_chunk(span<byte>(data2, kN2)));

  auto chunk = list.find_chunk(kN2 + 1);
  EXPECT_EQ(chunk.size(), kN1);
}

TEST(LlvmLibcFreeList, FindCanMoveUpThroughBuckets) {
  // Ensure that finding a chunk will move up through buckets if no appropriate
  // chunks were found in a given bucket
  FreeList<SIZE> list(example_sizes);
  constexpr size_t kN1 = 257;
  constexpr size_t kN2 = 513;

  byte data1[kN1] = {byte(0)};
  byte data2[kN2] = {byte(0)};

  // List should now be:
  // bkt[3] (257 bytes up to 512 bytes) -> 257 -> NULL
  // bkt[4] (513 bytes up to 1024 bytes) -> 513 -> NULL
  ASSERT_TRUE(list.add_chunk(span<byte>(data1, kN1)));
  ASSERT_TRUE(list.add_chunk(span<byte>(data2, kN2)));

  // Request a 300 byte chunk. This should return the 513 byte one
  auto chunk = list.find_chunk(kN1 + 1);
  EXPECT_EQ(chunk.size(), kN2);
}

TEST(LlvmLibcFreeList, RemoveUnknownChunkReturnsNotFound) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t N = 512;

  byte data[N] = {byte(0)};
  byte data2[N] = {byte(0)};

  ASSERT_TRUE(list.add_chunk(span<byte>(data, N)));
  EXPECT_FALSE(list.remove_chunk(span<byte>(data2, N)));
}

TEST(LlvmLibcFreeList, CanStoreMultipleChunksPerBucket) {
  FreeList<SIZE> list(example_sizes);
  constexpr size_t N = 512;

  byte data1[N] = {byte(0)};
  byte data2[N] = {byte(0)};

  ASSERT_TRUE(list.add_chunk(span<byte>(data1, N)));
  ASSERT_TRUE(list.add_chunk(span<byte>(data2, N)));

  auto chunk1 = list.find_chunk(N);
  ASSERT_TRUE(list.remove_chunk(chunk1));
  auto chunk2 = list.find_chunk(N);
  ASSERT_TRUE(list.remove_chunk(chunk2));

  // Ordering of the chunks doesn't matter
  EXPECT_TRUE(chunk1.data() != chunk2.data());
  EXPECT_TRUE(chunk1.data() == data1 || chunk1.data() == data2);
  EXPECT_TRUE(chunk2.data() == data1 || chunk2.data() == data2);
}
#endif

} // namespace LIBC_NAMESPACE_DECL
