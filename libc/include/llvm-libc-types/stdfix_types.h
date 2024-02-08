//===-- Definition from stdfix.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_STDFIX_TYPES_H
#define __LLVM_LIBC_MACROS_STDFIX_TYPES_H

#include <include/llvm-libc-macros/stdfix-macros.h>

#ifdef LIBC_COMPILER_HAS_FIXED_POINT
typedef _Fract fract;
typedef _Accum accum;
#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // __LLVM_LIBC_MACROS_STDFIX_TYPES_H
