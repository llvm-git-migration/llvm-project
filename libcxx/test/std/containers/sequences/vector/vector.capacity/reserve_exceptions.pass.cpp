//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// Check that std::vector::reserve provides strong exception guarantees

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../common.h"
#include "count_new.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_iterators.h"

template <typename T = int, typename Alloc = std::allocator<T>>
void test_allocation_exception(std::vector<T, Alloc>& v) {
  std::vector<T, Alloc> old_vector = v;
  T* old_data                      = v.data();
  std::size_t old_size             = v.size();
  std::size_t old_cap              = v.capacity();
  std::size_t new_cap              = v.max_size() + 1;
  try {
    v.reserve(new_cap);
  } catch (std::length_error&) {
    assert(v.size() == old_size);
    assert(v.capacity() == old_cap);
    assert(v.data() == old_data);
    assert(v == old_vector);
  }
}

template <typename T = int, typename Alloc = std::allocator<throwing_data<T>>>
void test_construction_exception(std::vector<throwing_data<T>, Alloc>& v, const std::vector<T>& in) {
  assert(v.empty() && !in.empty());
  int throw_after = 2 * in.size() - 1;
  v.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
    v.emplace_back(in[i], throw_after);

  throwing_data<T>* old_data = v.data();
  std::size_t old_size       = v.size();
  std::size_t old_cap        = v.capacity();
  std::size_t new_cap        = 2 * old_cap;

  try {
    v.reserve(new_cap);
  } catch (int) {
    assert(v.size() == old_size);
    assert(v.capacity() == old_cap);
    assert(v.data() == old_data);
    for (std::size_t i = 0; i < in.size(); ++i)
      assert(v[i].data_ == in[i]);
  }
}

void test_allocation_exceptions() {
  {
    std::vector<int> v;
    test_allocation_exception(v);
  }
  check_new_delete_called();

  {
    std::vector<int> v(10, 42);
    test_allocation_exception(v);
  }
  check_new_delete_called();

  {
    std::vector<int, min_allocator<int> > v(10, 42);
    test_allocation_exception(v);
  }
  check_new_delete_called();

  {
    std::vector<int, safe_allocator<int> > v(10, 42);
    test_allocation_exception(v);
  }
  check_new_delete_called();

  {
    std::vector<int, test_allocator<int> > v(10, 42);
    test_allocation_exception(v);
  }
  check_new_delete_called();

  {
    std::vector<int, limited_allocator<int, 100> > v(10, 42);
    test_allocation_exception(v);
  }
  check_new_delete_called();
}

void test_construction_exceptions() {
  {
    std::vector<throwing_data<int>> v;
    std::vector<int> in = {1, 2, 3, 4, 5};
    test_construction_exception(v, in);
  }
  check_new_delete_called();

  {
    std::vector<throwing_data<int>, min_allocator<throwing_data<int>>> v;
    std::vector<int> in = {1, 2, 3, 4, 5};
    test_construction_exception(v, in);
  }
  check_new_delete_called();

  {
    std::vector<throwing_data<int>, safe_allocator<throwing_data<int>> > v;
    std::vector<int> in(10, 42);
    test_construction_exception(v, in);
  }
  check_new_delete_called();

  {
    std::vector<throwing_data<int>, test_allocator<throwing_data<int>> > v;
    std::vector<int> in(10, 42);
    test_construction_exception(v, in);
  }
  check_new_delete_called();

  {
    std::vector<throwing_data<int>, limited_allocator<throwing_data<int>, 100> > v;
    std::vector<int> in(10, 42);
    test_construction_exception(v, in);
  }
  check_new_delete_called();
}

int main(int, char**) {
  test_allocation_exceptions();
  test_construction_exceptions();
}
