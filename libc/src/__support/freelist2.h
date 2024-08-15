//===-- Interface for freelist --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST2_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST2_H

#include "block.h"

namespace LIBC_NAMESPACE_DECL {

/// A circularly-linked FIFO list node storing a free Block. A list is a
/// FreeList2*; nullptr is an empty list. All Blocks on a list are the same
/// size.
class FreeList2 {
public:
  Block<> *block() const {
    return const_cast<Block<> *>(Block<>::from_usable_space(this));
  }

  /// @returns Size for all blocks on the list.
  size_t size() const { return block()->inner_size(); }

  /// Push to the back. The Block must be able to contain a FreeList2.
  static void push(FreeList2 *&list, Block<> *block);

  /// Pop the front.
  static void pop(FreeList2 *&list);

protected:
  /// Push an already-constructed node to the back.
  static void push(FreeList2 *&list, FreeList2 *node);

private:
  // Circularly linked pointers to adjacent nodes.
  FreeList2 *prev;
  FreeList2 *next;
};

LIBC_INLINE void FreeList2::push(FreeList2 *&list, Block<> *block) {
  LIBC_ASSERT(block->inner_size_free() >= sizeof(FreeList2) &&
              "block too small to accomodate free list node");
  push(list, new (block->usable_space()) FreeList2);
}

LIBC_INLINE void FreeList2::pop(FreeList2 *&list) {
  LIBC_ASSERT(list != nullptr && "cannot pop from empty list");
  if (list->next == list) {
    list = nullptr;
  } else {
    list->prev->next = list->next;
    list->next->prev = list->prev;
    list = list->next;
  }
}

LIBC_INLINE void FreeList2::push(FreeList2 *&list, FreeList2 *node) {
  if (list) {
    LIBC_ASSERT(Block<>::from_usable_space(node)->outer_size() ==
                    list->block()->outer_size() &&
                "freelist entries must have the same size");
    node->prev = list->prev;
    node->next = list;
    list->prev->next = node;
    list->prev = node;
  } else {
    list = node->prev = node->next = node;
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST2_H
