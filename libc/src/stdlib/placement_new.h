//===-- Interface for placement new ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_PLACEMENT_NEW_H
#define LLVM_LIBC_SRC_STDLIB_PLACEMENT_NEW_H

// FIXME: These should go inside new.h, but we can't use that header internally
// because it depends on defining aligned_alloc.
inline void *operator new(size_t, void *__p) { return __p; }
inline void *operator new[](size_t, void *__p) { return __p; }

#endif // LLVM_LIBC_SRC_STDLIB_PLACEMENT_NEW_H
