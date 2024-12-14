//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void flip();

#include <cassert>
#include <memory>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <typename Allocator = std::allocator<bool> >
TEST_CONSTEXPR_CXX20 void test_small_vector_flip(Allocator a) {
  bool b[] = {true, false, true};
  std::vector<bool, Allocator> v(b, b + 3, a);
  v.flip();
  assert(!v[0] && v[1] && !v[2]);
}

template <typename Allocator = std::allocator<bool> >
TEST_CONSTEXPR_CXX20 void test_large_vector_flip(Allocator a) {
  std::vector<bool, Allocator > v(1000, false, a);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = i & 1;
  std::vector<bool, Allocator > original = v;
  v.flip();
  for (size_t i = 0; i < v.size(); ++i)
    assert(v[i] == !original[i]);
  v.flip();
  assert(v == original);
}

TEST_CONSTEXPR_CXX20 bool tests() {
  // Test small vectors with different allocators
  test_small_vector_flip(std::allocator<bool>());
  test_small_vector_flip(min_allocator<bool>());
  test_small_vector_flip(test_allocator<bool>(5));

  // Test large vectors with different allocators
  test_large_vector_flip(std::allocator<bool>());
  test_large_vector_flip(min_allocator<bool>());
  test_large_vector_flip(test_allocator<bool>(5));

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif
  return 0;
}
