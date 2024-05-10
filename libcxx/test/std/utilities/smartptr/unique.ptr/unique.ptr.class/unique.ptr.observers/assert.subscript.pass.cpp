//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <memory>
//
// unique_ptr<T[]>
//
// T& operator[](std::size_t);

// This test ensures that we catch an out-of-bounds access in std::unique_ptr<T[]>::operator[]
// when unique_ptr has the appropriate ABI configuration.

#include <memory>

#include "check_assertion.h"

struct MyDeleter {
  void operator()(int* ptr) const { delete[] ptr; }
};

int main(int, char**) {
  // Check with a default deleter
  {
    {
      std::unique_ptr<int[]> ptr(new int[5]);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
    }

    {
      std::unique_ptr<int[]> ptr = std::make_unique<int[]>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
    }

#if TEST_STD_VER >= 20
    {
      std::unique_ptr<int[]> ptr = std::make_unique_for_overwrite<int[]>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
    }
#endif
  }

  // Check with a non-default deleter
#if defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
  {
    {
      std::unique_ptr<int[], MyDeleter> ptr = std::make_unique<int[]>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
    }

#  if TEST_STD_VER >= 20
    {
      std::unique_ptr<int[], MyDeleter> ptr = std::make_unique_for_overwrite<int[], MyDeleter>(5);
      TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
    }
#  endif
  }
#endif

  return 0;
}
