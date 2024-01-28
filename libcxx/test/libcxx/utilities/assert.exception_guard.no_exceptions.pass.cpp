//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// REQUIRES: libcpp-hardening-mode=debug
<<<<<<< HEAD
// XFAIL: availability-verbose_abort-missing
=======
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e
// ADDITIONAL_COMPILE_FLAGS: -fno-exceptions

#include <__utility/exception_guard.h>

#include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::__make_exception_guard([] {}), "__exception_guard not completed with exceptions disabled");
}
