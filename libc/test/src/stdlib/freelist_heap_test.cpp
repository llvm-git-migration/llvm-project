//===-- Unittests for freelist_heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/stdlib/freelist_heap.h"
#include "src/string/memcmp.h"
#include "src/string/memcpy.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

TEST(LlvmLibcFreeListHeap, CanAllocate) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr = allocator.Allocate(kAllocSize);

  ASSERT_NE(ptr, static_cast<void *>(nullptr));
  // In this case, the allocator should be returning us the start of the chunk.
  EXPECT_EQ(ptr, static_cast<void *>(
                     &buf[0] + FreeListHeap<>::BlockType::kBlockOverhead));
}

TEST(LlvmLibcFreeListHeap, AllocationsDontOverlap) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  void *ptr2 = allocator.Allocate(kAllocSize);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));

  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  uintptr_t ptr1_end = ptr1_start + kAllocSize;
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_GT(ptr2_start, ptr1_end);
}

TEST(LlvmLibcFreeListHeap, CanFreeAndRealloc) {
  // There's not really a nice way to test that Free works, apart from to try
  // and get that value back again.
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  allocator.Free(ptr1);
  void *ptr2 = allocator.Allocate(kAllocSize);

  EXPECT_EQ(ptr1, ptr2);
}

TEST(LlvmLibcFreeListHeap, ReturnsNullWhenAllocationTooLarge) {
  constexpr size_t N = 2048;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  EXPECT_EQ(allocator.Allocate(N), static_cast<void *>(nullptr));
}

TEST(LlvmLibcFreeListHeap, ReturnsNullWhenFull) {
  constexpr size_t N = 2048;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  EXPECT_NE(allocator.Allocate(N - FreeListHeap<>::BlockType::kBlockOverhead),
            static_cast<void *>(nullptr));
  EXPECT_EQ(allocator.Allocate(1), static_cast<void *>(nullptr));
}

TEST(LlvmLibcFreeListHeap, ReturnedPointersAreAligned) {
  constexpr size_t N = 2048;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(1);

  // Should be aligned to native pointer alignment
  uintptr_t ptr1_start = reinterpret_cast<uintptr_t>(ptr1);
  size_t alignment = alignof(void *);

  EXPECT_EQ(ptr1_start % alignment, static_cast<size_t>(0));

  void *ptr2 = allocator.Allocate(1);
  uintptr_t ptr2_start = reinterpret_cast<uintptr_t>(ptr2);

  EXPECT_EQ(ptr2_start % alignment, static_cast<size_t>(0));
}

TEST(LlvmLibcFreeListHeap, CanRealloc) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  constexpr size_t kNewAllocSize = 768;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(1)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  void *ptr2 = allocator.Realloc(ptr1, kNewAllocSize);

  ASSERT_NE(ptr1, static_cast<void *>(nullptr));
  ASSERT_NE(ptr2, static_cast<void *>(nullptr));
}

TEST(LlvmLibcFreeListHeap, ReallocHasSameContent) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = sizeof(int);
  constexpr size_t kNewAllocSize = sizeof(int) * 2;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(1)};
  // Data inside the allocated block.
  cpp::byte data1[kAllocSize];
  // Data inside the reallocated block.
  cpp::byte data2[kAllocSize];

  FreeListHeap<> allocator(buf);

  int *ptr1 = reinterpret_cast<int *>(allocator.Allocate(kAllocSize));
  *ptr1 = 42;
  memcpy(data1, ptr1, kAllocSize);
  int *ptr2 = reinterpret_cast<int *>(allocator.Realloc(ptr1, kNewAllocSize));
  memcpy(data2, ptr2, kAllocSize);

  ASSERT_NE(ptr1, static_cast<int *>(nullptr));
  ASSERT_NE(ptr2, static_cast<int *>(nullptr));
  // Verify that data inside the allocated and reallocated chunks are the same.
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(data1, data2, kAllocSize), 0);
}

TEST(LlvmLibcFreeListHeap, ReturnsNullReallocFreedPointer) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  constexpr size_t kNewAllocSize = 256;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  allocator.Free(ptr1);
  void *ptr2 = allocator.Realloc(ptr1, kNewAllocSize);

  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST(LlvmLibcFreeListHeap, ReallocSmallerSize) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  constexpr size_t kNewAllocSize = 256;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  void *ptr2 = allocator.Realloc(ptr1, kNewAllocSize);

  // For smaller sizes, Realloc will not shrink the block.
  EXPECT_EQ(ptr1, ptr2);
}

TEST(LlvmLibcFreeListHeap, ReallocTooLarge) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 512;
  constexpr size_t kNewAllocSize = 4096;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(0)};

  FreeListHeap<> allocator(buf);

  void *ptr1 = allocator.Allocate(kAllocSize);
  void *ptr2 = allocator.Realloc(ptr1, kNewAllocSize);

  // Realloc() will not invalidate the original pointer if Reallc() fails
  EXPECT_NE(static_cast<void *>(nullptr), ptr1);
  EXPECT_EQ(static_cast<void *>(nullptr), ptr2);
}

TEST(LlvmLibcFreeListHeap, CanCalloc) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 128;
  constexpr size_t kNum = 4;
  constexpr int size = kNum * kAllocSize;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(1)};
  constexpr cpp::byte zero{0};

  FreeListHeap<> allocator(buf);

  cpp::byte *ptr1 =
      reinterpret_cast<cpp::byte *>(allocator.Calloc(kNum, kAllocSize));

  // Calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST(LlvmLibcFreeListHeap, CanCallocWeirdSize) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 143;
  constexpr size_t kNum = 3;
  constexpr int size = kNum * kAllocSize;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(132)};
  constexpr cpp::byte zero{0};

  FreeListHeap<> allocator(buf);

  cpp::byte *ptr1 =
      reinterpret_cast<cpp::byte *>(allocator.Calloc(kNum, kAllocSize));

  // Calloc'd content is zero.
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(ptr1[i], zero);
  }
}

TEST(LlvmLibcFreeListHeap, CallocTooLarge) {
  constexpr size_t N = 2048;
  constexpr size_t kAllocSize = 2049;
  alignas(FreeListHeap<>::BlockType) cpp::byte buf[N] = {cpp::byte(1)};

  FreeListHeap<> allocator(buf);

  EXPECT_EQ(allocator.Calloc(1, kAllocSize), static_cast<void *>(nullptr));
}

} // namespace LIBC_NAMESPACE
