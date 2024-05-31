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
// TODO: We should probably have something akin to what scudo/sanitizer
// allocators do where each platform defines this.
constexpr size_t kSize = 0x40000000ULL; // 1GB
LIBC_CONSTINIT FreeListHeapBuffer<kSize> freelist_heap_buffer;
} // namespace

FreeListHeap<> *freelist_heap = &freelist_heap_buffer;

void *malloc(size_t size) { return freelist_heap->Allocate(size); }

void free(void *ptr) { freelist_heap->Free(ptr); }

void *calloc(size_t num, size_t size) {
  return freelist_heap->Calloc(num, size);
}

} // namespace LIBC_NAMESPACE
