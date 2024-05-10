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
#include <cstddef>
#include <string>

#include "check_assertion.h"

struct MyDeleter {
  void operator()(int* ptr) const { delete[] ptr; }
};

template <class WithCookie, class NoCookie>
void test() {
  // Check with a default deleter
  {
    // Types that have an array cookie
    {
      {
        std::unique_ptr<WithCookie[]> ptr(new WithCookie[5]);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
      {
        std::unique_ptr<WithCookie[]> ptr = std::make_unique<WithCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#if TEST_STD_VER >= 20
      {
        std::unique_ptr<WithCookie[]> ptr = std::make_unique_for_overwrite<WithCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#endif
    }

    // Types that don't have an array cookie (only available under the right ABI configuration)
#if defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
    {
      {
        std::unique_ptr<NoCookie[]> ptr(new NoCookie[5]);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
      {
        std::unique_ptr<NoCookie[]> ptr = std::make_unique<NoCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  if TEST_STD_VER >= 20
      {
        std::unique_ptr<NoCookie[]> ptr = std::make_unique_for_overwrite<NoCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  endif
    }
#endif // defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
  }

  // Check with a custom deleter (only available under the right ABI configuration)
#if defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
  {
    // Types that have an array cookie
    {
      {
        std::unique_ptr<WithCookie[], MyDeleter> ptr = std::make_unique<WithCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  if TEST_STD_VER >= 20
      {
        std::unique_ptr<WithCookie[], MyDeleter> ptr = std::make_unique_for_overwrite<WithCookie[], MyDeleter>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6], "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  endif
    }

    // Types that don't have an array cookie
    {
      {
        std::unique_ptr<NoCookie[], MyDeleter> ptr = std::make_unique<NoCookie[]>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  if TEST_STD_VER >= 20
      {
        std::unique_ptr<NoCookie[], MyDeleter> ptr = std::make_unique_for_overwrite<NoCookie[], MyDeleter>(5);
        TEST_LIBCPP_ASSERT_FAILURE(ptr[6] = 42, "unique_ptr<T[]>::operator[](index): index out of range");
      }
#  endif
    }
  }
#endif // defined(_LIBCPP_ABI_BOUNDED_UNIQUE_PTR)
}

template <std::size_t Size>
struct NoCookie {
  char padding[Size];
};

template <std::size_t Size>
struct WithCookie {
  ~WithCookie() {}
  char padding[Size];
};

int main(int, char**) {
  test<WithCookie<1>, NoCookie<1>>();
  test<WithCookie<2>, NoCookie<2>>();
  test<WithCookie<3>, NoCookie<3>>();
  test<WithCookie<4>, NoCookie<4>>();
  test<WithCookie<8>, NoCookie<8>>();
  test<WithCookie<16>, NoCookie<16>>();
  test<WithCookie<32>, NoCookie<32>>();
  test<WithCookie<256>, NoCookie<256>>();
  test<std::string, int>();

  return 0;
}
