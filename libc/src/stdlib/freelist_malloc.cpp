//===-- Implementation for freelist_malloc --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "freelist_heap.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"
#include "src/string/memcpy.h"
#include "src/string/memset.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {

namespace {
cpp::aligned_storage_t<sizeof(FreeListHeap<>), alignof(FreeListHeap<>)> buf;
} // namespace

FreeListHeap<> *freelist_heap;

// Define the global heap variables.
void MallocInit(uint8_t *heap_low_addr, uint8_t *heap_high_addr) {
  cpp::span<LIBC_NAMESPACE::cpp::byte> allocator_freelist_raw_heap =
      cpp::span<cpp::byte>(reinterpret_cast<cpp::byte *>(heap_low_addr),
                           heap_high_addr - heap_low_addr);
  freelist_heap = new (&buf) FreeListHeap<>(allocator_freelist_raw_heap);
}

void *malloc(size_t size) { return freelist_heap->Allocate(size); }

void free(void *ptr) { freelist_heap->Free(ptr); }

void *calloc(size_t num, size_t size) {
  return freelist_heap->Calloc(num, size);
}

} // namespace LIBC_NAMESPACE
