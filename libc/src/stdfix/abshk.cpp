//===-- Implementation of abshk function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "abshk.h"
#include "src/__support/common.h"
#include "src/__support/fixedpoint/fxbits.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(short accum, abshk, (short accum x)) {
  return fixedpoint::abs(x);
}

} // namespace LIBC_NAMESPACE
