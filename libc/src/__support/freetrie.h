//===-- Interface for freetrie
//--------------------------------------------===//freetrie.h
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
#define LLVM_LIBC_SRC___SUPPORT_FREETRIE_H

#include "freelist.h"

namespace LIBC_NAMESPACE_DECL {

/// A trie node containing a free list. The subtrie contains a contiguous
/// SizeRange of freelists.There is no relationship between the size of this
/// free list and the size ranges of the subtries.
class FreeTrie : public FreeList {
public:
  // Power-of-two range of sizes covered by a subtrie.
  struct SizeRange {
    size_t min;
    size_t width;

    constexpr SizeRange(size_t min, size_t width);

    /// @returns The lower half of the size range.
    SizeRange lower() const;

    /// @returns The lower half of the size range.
    SizeRange upper() const;

    /// @returns The largest size in this range.
    size_t max() const;

    /// @returns Whether the range contains the given size.
    /// Lower bound is inclusive, upper bound is exclusive.
    bool contains(size_t size) const;
  };

  struct InsertPos {
    FreeTrie *parent;
    FreeTrie **trie;
  };

  FreeTrie *&self();

  /// Push to the back of this node's free list.
  static void push(InsertPos pos, Block<> *block);

  /// Pop from the front of this node's free list.
  static void pop(FreeTrie *&trie);

  /// Finds the free trie for a given size. This may be a referance to a nullptr
  /// at the correct place in the trie structure. The caller must provide the
  /// SizeRange for this trie; the trie does not store it.
  static InsertPos find(FreeTrie *&trie, size_t size, SizeRange range);

  static FreeTrie **find_best_fit(FreeTrie *&trie, size_t size,
                                  SizeRange range);

private:
  /// Return an abitrary leaf.
  FreeTrie &leaf();

  // The child subtrie covering the lower half of this subtrie's size range.
  FreeTrie *lower;
  // The child subtrie covering the upper half of this subtrie's size range.
  FreeTrie *upper;

  FreeTrie *parent;
};

inline FreeTrie *&FreeTrie::self() {
  LIBC_ASSERT(parent && "reference in parent not defined on root");
  return parent->lower == this ? parent->lower : parent->upper;
}

LIBC_INLINE constexpr FreeTrie::SizeRange::SizeRange(size_t min, size_t width)
    : min(min), width(width) {
  LIBC_ASSERT(!(width & (width - 1)) && "width must be a power of two");
}

LIBC_INLINE FreeTrie::SizeRange FreeTrie::SizeRange::lower() const {
  return {min, width / 2};
}

LIBC_INLINE FreeTrie::SizeRange FreeTrie::SizeRange::upper() const {
  return {min + width / 2, width / 2};
}

LIBC_INLINE size_t FreeTrie::SizeRange::max() const {
  return min + (width - 1);
}

LIBC_INLINE bool FreeTrie::SizeRange::contains(size_t size) const {
  if (size < min)
    return false;
  if (size > min + width)
    return false;
  return true;
}

LIBC_INLINE void FreeTrie::push(InsertPos pos, Block<> *block) {
  LIBC_ASSERT(block->inner_size_free() >= sizeof(FreeTrie) &&
              "block too small to accomodate free trie node");
  FreeTrie *node = new (block->usable_space()) FreeTrie;
  // The trie links are irrelevant for all but the first node in the free list.
  if (!*pos.trie) {
    node->lower = node->upper = nullptr;
    node->parent = pos.parent;
  }
  FreeList *list = *pos.trie;
  FreeList::push(list, node);
  *pos.trie = static_cast<FreeTrie *>(list);
}

LIBC_INLINE void FreeTrie::pop(FreeTrie *&trie) {
  FreeList *list = trie;
  FreeList::pop(list);
  FreeTrie *new_trie = static_cast<FreeTrie *>(list);
  if (new_trie) {
    // The freelist is non-empty, so copy the trie links to the new head.
    new_trie->lower = trie->lower;
    new_trie->upper = trie->upper;
    new_trie->parent = trie->parent;
    trie = new_trie;
    return;
  }

  // The freelist is empty.

  FreeTrie &l = trie->leaf();
  if (&l == trie) {
    // If the root is a leaf, then removing it empties the trie.
    trie = nullptr;
    return;
  }

  // Replace the root with an arbitrary leaf. This is legal because there is
  // no relationship between the size of the root and its children.
  l.lower = trie->lower;
  l.upper = trie->upper;
  l.parent = trie->parent;
  trie = &l;
}

LIBC_INLINE FreeTrie::InsertPos FreeTrie::find(FreeTrie *&trie, size_t size,
                                               SizeRange range) {
  LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
  InsertPos pos = {nullptr, &trie};
  while (*pos.trie && (*pos.trie)->size() != size) {
    LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
    pos.parent = *pos.trie;
    if (range.lower().contains(size)) {
      pos.trie = &(*pos.trie)->lower;
      range = range.lower();
    } else {
      pos.trie = &(*pos.trie)->upper;
      range = range.upper();
    }
  }
  return pos;
}

LIBC_INLINE FreeTrie **FreeTrie::find_best_fit(FreeTrie *&trie, size_t size,
                                               SizeRange range) {
  if (!trie)
    return nullptr;

  LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
  FreeTrie **cur = &trie;
  FreeTrie **best_fit = nullptr;
  FreeTrie **deferred_upper_trie = nullptr;
  SizeRange deferred_upper_range{0, 0};

  // Inductively assume all better fits than the current best are in the
  // current subtrie.
  while (true) {
    LIBC_ASSERT(range.max() >= size && "range could not fit requested size");

    // If the current node is an exact fit, it is a best fit.
    if ((*cur)->size() == size)
      return cur;

    if ((*cur)->size() > size &&
        (!best_fit || (*cur)->size() < (*best_fit)->size())) {
      // The current node is a better fit.
      best_fit = cur;
      LIBC_ASSERT(
          !deferred_upper_trie ||
          deferred_upper_range.min > (*cur)->size() &&
              "deferred upper subtrie should be outclassed by new best fit");
      deferred_upper_trie = nullptr;
    }

    // Determine which subtries might contain better fits.
    bool lower_impossible = !(*cur)->lower || range.lower().max() < size;
    bool upper_impossible =
        !(*cur)->upper ||
        (best_fit && range.upper().min >= (*best_fit)->size());

    if (lower_impossible && upper_impossible) {
      if (!deferred_upper_trie)
        return best_fit;
      // Scan the deferred upper subtrie and consider whether any element within
      // provides a better fit.
      //
      // This can only ever be reached once. In a deferred upper subtrie, every
      // node fits, so the scan can always summarily ignore an upper suptrie
      // rather than deferring it.
      cur = deferred_upper_trie;
      range = deferred_upper_range;
      deferred_upper_trie = nullptr;
      continue;
    }

    if (lower_impossible) {
      cur = &(*cur)->upper;
      range = range.upper();
    } else if (upper_impossible) {
      cur = &(*cur)->lower;
      range = range.lower();
    } else {
      // Both subtries might contain a better fit. Any fit in the lower subtrie
      // is better than the any fit in the upper subtrie, so scan the lower
      // subtrie and return to the upper one if necessary.
      cur = &(*cur)->lower;
      range = range.lower();
      LIBC_ASSERT(!deferred_upper_trie ||
                  range.upper().max() < deferred_upper_range.min &&
                      "old deferred upper subtrie should be outclassed by new");
      deferred_upper_trie = &(*cur)->upper;
    }
  }
}

LIBC_INLINE FreeTrie &FreeTrie::leaf() {
  FreeTrie *cur = this;
  while (cur->lower || cur->upper)
    cur = cur->lower ? cur->lower : cur->upper;
  return *cur;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
