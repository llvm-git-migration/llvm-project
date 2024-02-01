//===-- Invasive queue implementation. --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An invasive list that implements the insque and remque semantics.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIBC_SRC___SUPPORT_INVASIVE_QUEUE_H
#define LLVM_LIBC_SRC___SUPPORT_INVASIVE_QUEUE_H

#include "common.h"

namespace LIBC_NAMESPACE {
namespace internal {

struct InvasiveList {
  struct InvasiveNodeHeader {
    InvasiveNodeHeader *next;
    InvasiveNodeHeader *prev;
  };

  LIBC_INLINE static void insert(InvasiveNodeHeader *elem, InvasiveNodeHeader *prev) {
    if (!prev) {
      // The list is linear and elem will be the only element.
      elem->next = nullptr;
      elem->prev = nullptr;
      return;
    }

    auto next = prev->next;

    elem->next = next;
    elem->prev = prev;

    prev->next = elem;
    if (next)
      next->prev = elem;
  }

  LIBC_INLINE static void remove(InvasiveNodeHeader *elem) {
    auto prev = elem->prev;
    auto next = elem->next;

    if (prev)
      prev->next = next;
    if (next)
      next->prev = prev;
  }
};

} // namespace internal
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_INVASIVE_QUEUE_H
