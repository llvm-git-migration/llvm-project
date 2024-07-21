//===-- Implementation of asprintf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/asprintf.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/macros/config.h"
#include "src/stdio/vasprintf.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, asprintf,
                   (char **__restrict buffer, const char *format, ...)) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vasprintf(buffer, format, vlist);
  va_end(vlist);
  return ret;
}

} // namespace LIBC_NAMESPACE
