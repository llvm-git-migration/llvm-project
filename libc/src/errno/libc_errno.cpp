//===-- Implementation of errno -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE {

struct Errno {
  void operator=(int);
  operator int();
};

} // namespace LIBC_NAMESPACE

#ifdef LIBC_TARGET_ARCH_IS_GPU
// If we are targeting the GPU we currently don't support 'errno'. We simply
// consume it.
void LIBC_NAMESPACE::Errno::operator=(int) {}
LIBC_NAMESPACE::Errno::operator int() { return 0; }

#elif !defined(LIBC_COPT_PUBLIC_PACKAGING)
// This mode is for unit testing.  We just use another internal errno.
LIBC_THREAD_LOCAL int __llvmlibc_internal_errno;

void LIBC_NAMESPACE::Errno::operator=(int a) { __llvmlibc_internal_errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return __llvmlibc_internal_errno; }

#elif defined(LIBC_FULL_BUILD)
// This mode is for public libc archive, hermetic, and integration tests.
// In full build mode, we provide all the errno storage ourselves.
extern "C" {
LIBC_THREAD_LOCAL int __llvmlibc_errno;
}

void LIBC_NAMESPACE::Errno::operator=(int a) { __llvmlibc_errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return __llvmlibc_errno; }

#else
// In overlay mode, we simply use the system errno.
#include <errno.h>

void LIBC_NAMESPACE::Errno::operator=(int a) { errno = a; }
LIBC_NAMESPACE::Errno::operator int() { return errno; }

#endif // LIBC_FULL_BUILD

// Define the global `libc_errno` instance.
LIBC_NAMESPACE::Errno libc_errno;
