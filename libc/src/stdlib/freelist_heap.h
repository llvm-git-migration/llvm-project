//===-- Interface for freelist_heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H
#define LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H

#include <stddef.h>

#include "block.h"
#include "freelist.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/string/memcpy.h"
#include "src/string/memset.h"

namespace LIBC_NAMESPACE {

void MallocInit(uint8_t *heap_low_addr, uint8_t *heap_high_addr);

using cpp::optional;
using cpp::span;

static constexpr cpp::array<size_t, 6> defaultBuckets{16,  32,  64,
                                                      128, 256, 512};

template <size_t kNumBuckets = defaultBuckets.size()> class FreeListHeap {
public:
  using BlockType = Block<>;

  template <size_t> friend class FreeListHeapBuffer;

  struct HeapStats {
    size_t total_bytes;
    size_t bytes_allocated;
    size_t cumulative_allocated;
    size_t cumulative_freed;
    size_t total_allocate_calls;
    size_t total_free_calls;
  };
  FreeListHeap(span<cpp::byte> region);

  void *Allocate(size_t size);
  void Free(void *ptr);
  void *Realloc(void *ptr, size_t size);
  void *Calloc(size_t num, size_t size);

  void LogHeapStats();
  const HeapStats &heap_stats() const { return heap_stats_; }

private:
  span<cpp::byte> BlockToSpan(BlockType *block) {
    return span<cpp::byte>(block->usable_space(), block->inner_size());
  }

  void InvalidFreeCrash() { __builtin_trap(); }

  span<cpp::byte> region_;
  FreeList<kNumBuckets> freelist_;
  HeapStats heap_stats_;
};

template <size_t kNumBuckets>
FreeListHeap<kNumBuckets>::FreeListHeap(span<cpp::byte> region)
    : freelist_(defaultBuckets), heap_stats_() {
  auto result = BlockType::init(region);
  BlockType *block = *result;

  freelist_.AddChunk(BlockToSpan(block));

  region_ = region;
  heap_stats_.total_bytes = region.size();
}

template <size_t kNumBuckets>
void *FreeListHeap<kNumBuckets>::Allocate(size_t size) {
  // Find a chunk in the freelist. Split it if needed, then return

  auto chunk = freelist_.FindChunk(size);

  if (chunk.data() == nullptr) {
    return nullptr;
  }
  freelist_.RemoveChunk(chunk);

  BlockType *chunk_block = BlockType::from_usable_space(chunk.data());

  // Split that chunk. If there's a leftover chunk, add it to the freelist
  optional<BlockType *> result = BlockType::split(chunk_block, size);
  if (result) {
    freelist_.AddChunk(BlockToSpan(*result));
  }

  chunk_block->mark_used();

  heap_stats_.bytes_allocated += size;
  heap_stats_.cumulative_allocated += size;
  heap_stats_.total_allocate_calls += 1;

  return chunk_block->usable_space();
}

template <size_t kNumBuckets> void FreeListHeap<kNumBuckets>::Free(void *ptr) {
  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  if (bytes < region_.data() || bytes >= region_.data() + region_.size()) {
    InvalidFreeCrash();
    return;
  }

  BlockType *chunk_block = BlockType::from_usable_space(bytes);

  size_t size_freed = chunk_block->inner_size();
  // Ensure that the block is in-use
  if (!chunk_block->used()) {
    InvalidFreeCrash();
    return;
  }
  chunk_block->mark_free();
  // Can we combine with the left or right blocks?
  BlockType *prev = chunk_block->prev();
  BlockType *next = nullptr;

  if (!chunk_block->last()) {
    next = chunk_block->next();
  }

  if (prev != nullptr && !prev->used()) {
    // Remove from freelist and merge
    freelist_.RemoveChunk(BlockToSpan(prev));
    chunk_block = chunk_block->prev();
    BlockType::merge_next(chunk_block);
  }

  if (next != nullptr && !next->used()) {
    freelist_.RemoveChunk(BlockToSpan(next));
    BlockType::merge_next(chunk_block);
  }
  // Add back to the freelist
  freelist_.AddChunk(BlockToSpan(chunk_block));

  heap_stats_.bytes_allocated -= size_freed;
  heap_stats_.cumulative_freed += size_freed;
  heap_stats_.total_free_calls += 1;
}

// Follows constract of the C standard realloc() function
// If ptr is free'd, will return nullptr.
template <size_t kNumBuckets>
void *FreeListHeap<kNumBuckets>::Realloc(void *ptr, size_t size) {
  if (size == 0) {
    Free(ptr);
    return nullptr;
  }

  // If the pointer is nullptr, allocate a new memory.
  if (ptr == nullptr) {
    return Allocate(size);
  }

  cpp::byte *bytes = static_cast<cpp::byte *>(ptr);

  // TODO(chenghanzh): Enhance with debug information for out-of-range and more.
  if (bytes < region_.data() || bytes >= region_.data() + region_.size()) {
    return nullptr;
  }

  BlockType *chunk_block = BlockType::from_usable_space(bytes);
  if (!chunk_block->used()) {
    return nullptr;
  }
  size_t old_size = chunk_block->inner_size();

  // Do nothing and return ptr if the required memory size is smaller than
  // the current size.
  if (old_size >= size) {
    return ptr;
  }

  void *new_ptr = Allocate(size);
  // Don't invalidate ptr if Allocate(size) fails to initilize the memory.
  if (new_ptr == nullptr) {
    return nullptr;
  }
  memcpy(new_ptr, ptr, old_size);

  Free(ptr);
  return new_ptr;
}

template <size_t kNumBuckets>
void *FreeListHeap<kNumBuckets>::Calloc(size_t num, size_t size) {
  void *ptr = Allocate(num * size);
  if (ptr != nullptr) {
    memset(ptr, 0, num * size);
  }
  return ptr;
}

extern FreeListHeap<> *freelist_heap;

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_FREELIST_HEAP_H
