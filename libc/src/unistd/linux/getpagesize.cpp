//===-- Implementation of getpagesize -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpagesize.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/page_size.h"
#include "src/sys/auxv/getauxval.h"
#include "src/unistd/sysconf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpagesize, ()) {
#if LIBC_PAGE_SIZE == LIBC_PAGE_SIZE_SYSTEM
  int r = (int)getauxval(AT_PAGESZ);
  if (r == 0)
    return (int)sysconf(_SC_PAGESIZE);
  return r;
#else
  return LIBC_PAGE_SIZE;
#endif
}

} // namespace LIBC_NAMESPACE_DECL
