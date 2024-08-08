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
  class SizeRange {
  public:
    SizeRange(size_t min, size_t width);

    /// @returns The lower half of the size range.
    SizeRange lower() const;

    /// @returns The lower half of the size range.
    SizeRange upper() const;

    /// @returns The split point between lower and upper.
    size_t middle() const;

    bool contains(size_t size) const;

  private:
    size_t min;
    size_t width;
  };

  /// Push to the back of this node's free list.
  static void push(FreeTrie *&trie, Block<> *block);

  /// Pop from the front of this node's free list.
  static void pop(FreeTrie *&trie);

  /// Finds the free trie for a given size. This may be a referance to a nullptr
  /// at the correct place in the trie structure. The caller must provide the
  /// SizeRange for this trie; the trie does not store it.
  static FreeTrie *&find(FreeTrie *&trie, size_t size, SizeRange range);

  // The containing trie or nullptr if this is the root.
  FreeTrie *parent;
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
  return {middle(), width / 2};
}

LIBC_INLINE size_t FreeTrie::SizeRange::middle() const {
  return min + width / 2;
}

LIBC_INLINE bool FreeTrie::SizeRange::contains(size_t size) const {
  if (size < min)
    return false;
  if (size > min + width)
    return false;
  return true;
}

LIBC_INLINE void FreeTrie::push(FreeTrie *&trie, Block<> *block) {
  LIBC_ASSERT(block->inner_size() >= sizeof(FreeTrie) &&
              "block too small to accomodate free trie node");
  FreeTrie *node = new (block->usable_space()) FreeTrie;
  // The trie links are irrelevant for all but the first node in the free list.
  if (!trie)
    node->parent = node->lower = node->upper = nullptr;
  FreeList2 *list = trie;
  FreeList2::push(list, node);
  trie = static_cast<FreeTrie *>(list);
}

LIBC_INLINE void FreeTrie::pop(FreeTrie *&trie) {
  FreeList2 *list = trie;
  FreeList2::pop(list);
  FreeTrie *new_trie = static_cast<FreeTrie *>(list);
  if (new_trie) {
    new_trie->parent = trie->parent;
    new_trie->lower = trie->lower;
    new_trie->upper = trie->upper;
  } else {
    // TODO
  }
  trie = new_trie;
}

FreeTrie *&FreeTrie::find(FreeTrie *&trie, size_t size, SizeRange range) {
  LIBC_ASSERT(range.contains(size) && "requested size out of trie range");
  if (!trie || trie->size() == size)
    return trie;
  return find(size <= range.middle() ? trie->lower : trie->upper, size,
              size <= range.middle() ? range.lower() : range.upper());
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
