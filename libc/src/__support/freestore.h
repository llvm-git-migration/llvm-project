//===-- Interface for freestore ------------------------------------------===//
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
  void set_range(FreeTrie::SizeRange range);
  void insert(Block<> *block);
  Block<> *remove_best_fit(size_t size);

private:
  static constexpr size_t MIN_SIZE = sizeof(FreeList2);
  static constexpr size_t MIN_LARGE_SIZE = sizeof(FreeTrie);
  static constexpr size_t NUM_SMALL_SIZES =
      (align_up(MIN_LARGE_SIZE) - align_up(MIN_SIZE)) / alignof(max_align_t);

  static bool is_small(Block<> *block);
  static bool is_small(size_t size);

  FreeList2 *&small_list(Block<> *block);
  FreeList2 *&small_list(size_t size);

  cpp::array<FreeList2 *, NUM_SMALL_SIZES> small_lists = {nullptr};
  FreeTrie *large_trie = nullptr;
  FreeTrie::SizeRange range = {0, 0};
};

inline void FreeStore::set_range(FreeTrie::SizeRange range) {
  LIBC_ASSERT(!large_trie && "cannot change the range of a preexisting trie");
  this->range = range;
}

inline void FreeStore::insert(Block<> *block) {
  if (block->inner_size_free() < MIN_SIZE)
    return;
  if (is_small(block))
    FreeList2::push(small_list(block), block);
  else
    FreeTrie::push(FreeTrie::find(large_trie, block->inner_size(), range),
                   block);
}

inline Block<> *FreeStore::remove_best_fit(size_t size) {
  if (is_small(size)) {
    for (FreeList2 *&list : small_lists) {
      if (!list || list->size() < size)
        continue;
      Block<> *block = list->block();
      FreeList2::pop(list);
      return block;
    }
    return nullptr;
  } else {
    FreeTrie **best_fit = FreeTrie::find_best_fit(large_trie, size, range);
    if (!best_fit)
      return nullptr;
    Block<> *block = (*best_fit)->block();
    FreeTrie::pop(*best_fit);
    return block;
  }
}

inline bool FreeStore::is_small(Block<> *block) {
  return block->inner_size_free() < MIN_LARGE_SIZE;
}

inline bool FreeStore::is_small(size_t size) {
  if (size < sizeof(Block<>::offset_type))
    return true;
  return size - sizeof(Block<>::offset_type) < MIN_LARGE_SIZE;
}

inline FreeList2 *&FreeStore::small_list(Block<> *block) {
  LIBC_ASSERT(is_small(block) && "only legal for small blocks");
  return small_lists[(block->inner_size_free() - MIN_SIZE) /
                     alignof(max_align_t)];
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREESTORE_H
