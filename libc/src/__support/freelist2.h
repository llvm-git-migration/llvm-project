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

/// A FIFO free-list storing Blocks of the same size.
class FreeList2 {
protected:
  // A circular doubly-linked node.
  struct Node {
    Node *prev;
    Node *next;
  };

public:
  // Blocks with inner sizes smaller than this must not be pushed.
  static constexpr size_t MIN_INNER_SIZE = sizeof(Node);

  bool empty() const { return !begin_; }
  Block<> *front() const;

  /// Push to the back.
  void push(Block<> *block);

  /// Pop the front.
  void pop();

private:
  Node *begin_ = nullptr;
};

LIBC_INLINE Block<> *FreeList2::front() const {
  LIBC_ASSERT(!empty());
  return Block<>::from_usable_space(begin_);
}

LIBC_INLINE void FreeList2::push(Block<> *block) {
  LIBC_ASSERT(block->inner_size() >= MIN_INNER_SIZE &&
              "block too small to accomodate free list node");
  Node *node = new (block->usable_space()) Node;
  if (begin_) {
    LIBC_ASSERT(block->outer_size() == front()->outer_size() &&
                "freelist entries must have the same size");
    node->prev = begin_->prev;
    node->next = begin_;
    begin_->prev->next = node;
    begin_->prev = node;
  } else {
    begin_ = node->prev = node->next = node;
  }
}

LIBC_INLINE void FreeList2::pop() {
  LIBC_ASSERT(!empty());
  if (begin_->next == begin_) {
    begin_ = nullptr;
  } else {
    begin_->prev->next = begin_->next;
    begin_->next->prev = begin_->prev;
    begin_ = begin_->next;
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST2_H
