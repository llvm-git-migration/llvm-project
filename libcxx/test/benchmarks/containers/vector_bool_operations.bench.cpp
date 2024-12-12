//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "ContainerBenchmarks.h"
#include "../GenerateInput.h"

using namespace ContainerBenchmarks;

template <typename T, typename SIZE_TYPE = std::size_t, typename DIFF_TYPE = std::ptrdiff_t>
class CustomSizedAllocator {
  template <typename U, typename Sz, typename Diff>
  friend class CustomSizedAllocator;

public:
  using value_type                  = T;
  using size_type                   = SIZE_TYPE;
  using difference_type             = DIFF_TYPE;
  using propagate_on_container_swap = std::true_type;

  explicit CustomSizedAllocator(int i = 0) : data_(i) {}

  template <typename U, typename Sz, typename Diff>
  constexpr CustomSizedAllocator(const CustomSizedAllocator<U, Sz, Diff>& a) noexcept : data_(a.data_) {}

  constexpr T* allocate(size_type n) {
    if (n > max_size())
      throw std::bad_array_new_length();
    return std::allocator<T>().allocate(n);
  }

  constexpr void deallocate(T* p, size_type n) noexcept { std::allocator<T>().deallocate(p, n); }

  constexpr size_type max_size() const noexcept { return std::numeric_limits<size_type>::max() / sizeof(value_type); }

  int get() { return data_; }

private:
  int data_;

  constexpr friend bool operator==(const CustomSizedAllocator& a, const CustomSizedAllocator& b) {
    return a.data_ == b.data_;
  }
  constexpr friend bool operator!=(const CustomSizedAllocator& a, const CustomSizedAllocator& b) {
    return a.data_ != b.data_;
  }
};

BENCHMARK_CAPTURE(BM_Move_Assignment,
                  vector_bool_uint32_t,
                  std::vector<bool, CustomSizedAllocator<bool, std::uint32_t, std::int32_t>>{},
                  CustomSizedAllocator<bool, std::uint32_t, std::int32_t>{})
    ->Arg(5140480);

BENCHMARK_CAPTURE(BM_Move_Assignment,
                  vector_bool_uint64_t,
                  std::vector<bool, CustomSizedAllocator<bool, std::uint64_t, std::int64_t>>{},
                  CustomSizedAllocator<bool, std::uint64_t, std::int64_t>{})
    ->Arg(5140480);

BENCHMARK_MAIN();
