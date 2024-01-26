//===-- Implementation header for errno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERRNO_LIBC_ERRNO_H
#define LLVM_LIBC_SRC_ERRNO_LIBC_ERRNO_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

#include <errno.h>

// This header is to be consumed by internal implementations, in which all of
// them should refer to `libc_errno` instead of using `errno` directly from
// <errno.h> header.

namespace LIBC_NAMESPACE {
struct Errno {
  void operator=(int);
  operator int();
};
} // namespace LIBC_NAMESPACE

extern LIBC_NAMESPACE::Errno libc_errno;

#endif // LLVM_LIBC_SRC_ERRNO_LIBC_ERRNO_H
