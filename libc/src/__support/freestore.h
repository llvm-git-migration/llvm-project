//===-- Interface for freestore -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
#define LLVM_LIBC_SRC___SUPPORT_FREESTORE_H

#include "freetrie.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE static constexpr size_t align_up(size_t value) {
  constexpr size_t ALIGNMENT = alignof(max_align_t);
  return (value + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

class FreeStore {
public:
  FreeStore(Block<> *block);

private:
  static constexpr size_t MIN_SIZE = sizeof(FreeList2);
  static constexpr size_t MIN_LARGE_SIZE = sizeof(FreeTrie);
  static constexpr size_t NUM_SMALL_SIZES =
      (align_up(MIN_LARGE_SIZE) - align_up(MIN_SIZE)) / alignof(max_align_t);

  static bool is_small(Block<> *block);

  FreeList2 *&small_list(Block<> *block);

  cpp::array<FreeList2 *, NUM_SMALL_SIZES> small_lists = {nullptr};
  FreeTrie *large_trie = nullptr;
};

inline FreeStore::FreeStore(Block<> *block) {
  if (is_small(block))
    FreeList2::push(small_list(block), block);
  else
    FreeTrie::push(large_trie, block);
}

bool FreeStore::is_small(Block<> *block) {
  return block->inner_size_free() <= MIN_LARGE_SIZE;
}

FreeList2 *&FreeStore::small_list(Block<> *block) {
  LIBC_ASSERT(is_small(block) && "can legal for small blocks");
  return small_lists[block->inner_size_free() / alignof(max_align_t)];
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
