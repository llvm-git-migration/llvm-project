//===-- Interface for freetrie --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
#define LLVM_LIBC_SRC___SUPPORT_FREETRIE_H

#include "freelist2.h"

namespace LIBC_NAMESPACE_DECL {

/// A trie node containing a free list. The subtrie contains a contiguous
/// SizeRange of freelists.There is no relationship between the size of this
/// free list and the size ranges of the subtries.
class FreeTrie : public FreeList2 {
public:
  // Power-of-two range of sizes covered by a subtrie.
  struct SizeRange {
    size_t min;
    size_t width;

    SizeRange(size_t min, size_t width);

    /// @returns The lower half of the size range.
    SizeRange lower() const;

    /// @returns The lower half of the size range.
    SizeRange upper() const;

    /// @returns Whether the range contains the given size.
    /// Lower bound is inclusive, upper bound is exclusive.
    bool contains(size_t size) const;
  };

  /// Push to the back of this node's free list.
  static void push(FreeTrie *&trie, Block<> *block);

  /// Pop from the front of this node's free list.
  static void pop(FreeTrie *&trie);

  /// Finds the free trie for a given size. This may be a referance to a nullptr
  /// at the correct place in the trie structure. The caller must provide the
  /// SizeRange for this trie; the trie does not store it.
  static FreeTrie *&find(FreeTrie *&trie, size_t size, SizeRange range);

  static FreeTrie **find_best_fit(FreeTrie *&trie, size_t size,
                                  SizeRange range);

private:
  /// Return the smallest-sized free list in the trie.
  static FreeTrie *&smallest(FreeTrie *&trie);

  /// Return an abitrary leaf.
  FreeTrie &leaf();

  // The child subtrie covering the lower half of this subtrie's size range.
  FreeTrie *lower;
  // The child subtrie covering the upper half of this subtrie's size range.
  FreeTrie *upper;
};

LIBC_INLINE FreeTrie::SizeRange::SizeRange(size_t min, size_t width)
    : min(min), width(width) {
  LIBC_ASSERT(!(width & (width - 1)) && "width must be a power of two");
}

LIBC_INLINE FreeTrie::SizeRange FreeTrie::SizeRange::lower() const {
  return {min, width / 2};
}

LIBC_INLINE FreeTrie::SizeRange FreeTrie::SizeRange::upper() const {
  return {min + width / 2, width / 2};
}

LIBC_INLINE bool FreeTrie::SizeRange::contains(size_t size) const {
  if (size < min)
    return false;
  if (size > min + width)
    return false;
  return true;
}

LIBC_INLINE void FreeTrie::push(FreeTrie *&trie, Block<> *block) {
  LIBC_ASSERT(block->inner_size_free() >= sizeof(FreeTrie) &&
              "block too small to accomodate free trie node");
  FreeTrie *node = new (block->usable_space()) FreeTrie;
  // The trie links are irrelevant for all but the first node in the free list.
  if (!trie)
    node->lower = node->upper = nullptr;
  FreeList2 *list = trie;
  FreeList2::push(list, node);
  trie = static_cast<FreeTrie *>(list);
}

LIBC_INLINE void FreeTrie::pop(FreeTrie *&trie) {
  FreeList2 *list = trie;
  FreeList2::pop(list);
  FreeTrie *new_trie = static_cast<FreeTrie *>(list);
  if (new_trie) {
    // The freelist is non-empty, so copy the trie links to the new head.
    new_trie->lower = trie->lower;
    new_trie->upper = trie->upper;
    trie = new_trie;
    return;
  }

  // The freelist is empty.

  FreeTrie &l = trie->leaf();
  if (&l == trie) {
    // The last element of the trie was remved.
    trie = nullptr;
    return;
  }

  // Replace the root with an arbitrary leaf. This is legal because there is
  // no relationship between the size of the root and its children.
  l.lower = trie->lower;
  l.upper = trie->upper;
  trie = &l;
}

FreeTrie *&FreeTrie::find(FreeTrie *&trie, size_t size, SizeRange range) {
  LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
  FreeTrie **cur = &trie;
  while (*cur && (*cur)->size() != size) {
    LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
    if (range.lower().contains(size)) {
      cur = &(*cur)->lower;
      range = range.lower();
    } else {
      cur = &(*cur)->upper;
      range = range.upper();
    }
  }
  return *cur;
}

FreeTrie **FreeTrie::find_best_fit(FreeTrie *&trie, size_t size,
                                   SizeRange range) {
  LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
  FreeTrie **cur = &trie;
  FreeTrie **skipped_upper_trie = nullptr;

  // Inductively assume the best fit is in this subtrie.
  while (*cur) {
    LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
    size_t cur_size = (*cur)->size();
    if (cur_size == size)
      return cur;
    if (range.lower().contains(size)) {
      // If the lower subtree has at least one entry >= size, the best fit is in
      // the lower subtrie. But if the lower subtrie contains only smaller
      // sizes, the best fit is in the larger trie. So keep track of it.
      if ((*cur)->upper)
        skipped_upper_trie = &(*cur)->upper;
      cur = &(*cur)->lower;
      range = range.lower();
    } else {
      // The lower child is too small, so the best fit is in the upper subtrie.
      cur = &(*cur)->upper;
      range = range.upper();
    }
  }

  // A lower subtrie contained size in its range, but it had only entries
  // smaller than size. Accordingly, the best fit is the smallest entry in the
  // corresponding upper subtrie.
  return &FreeTrie::smallest(*skipped_upper_trie);
}

FreeTrie &FreeTrie::leaf() {
  FreeTrie *cur = this;
  while (cur->lower || cur->upper)
    cur = cur->lower ? cur->lower : cur->upper;
  return *cur;
}

FreeTrie *&FreeTrie::smallest(FreeTrie *&trie) {
  FreeTrie **cur = &trie;
  FreeTrie **ret = nullptr;
  while (*cur) {
    if (!ret || (*cur)->size() < (*ret)->size())
      ret = cur;
    cur = (*cur)->lower ? &(*cur)->lower : &(*cur)->upper;
  }
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
