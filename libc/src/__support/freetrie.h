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

/// A trie representing a map of free lists covering a contiguous SizeRange.
class FreeTrie : public FreeList2 {
private:
  // A subtrie of free lists covering a contiguous SizeRange. This is also a
  // free list with a size somewhere within the range. There is no relationship
  // between the size of this free list and the sizes of the lower and upper
  // subtries.
  struct Node : public FreeList2::Node {
    // The containing trie or nullptr if this is the root.
    Node *parent;
    // The child subtrie covering the lower half of this subtrie's size range.
    Node *lower;
    // The child subtrie covering the upper half of this subtrie's size range.
    Node *upper;
  };

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

  private:
    size_t min;
    size_t width;
  };

  // Blocks with inner sizes smaller than this must not be pushed.
  static constexpr size_t MIN_INNER_SIZE = sizeof(Node);

  /// Push to this freelist.
  void push(Block<> *block);

  /// Push to the correctly-sized freelist. The caller must provide the size
  /// range for this trie, since it isn't stored within.
  void push(Block<> *block, SizeRange range);
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

LIBC_INLINE void FreeTrie::push(Block<> *block) {
  LIBC_ASSERT(block->inner_size() >= MIN_INNER_SIZE &&
              "block too small to accomodate free list node");
  Node *node = new (block->usable_space()) Node;
  if (empty())
    node->parent = node->lower = node->upper = nullptr;
  FreeList2::push(node);
}

LIBC_INLINE void FreeTrie::push(Block<> *block, SizeRange range) {
  if (empty() || block->outer_size() == front()->outer_size()) {
    push(block);
    return;
  }

  // TODO;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREETRIE_H
