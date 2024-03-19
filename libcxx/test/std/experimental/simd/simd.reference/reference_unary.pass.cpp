//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.reference]
// reference operator++() && noexcept;
// value_type operator++(int) && noexcept;
// reference operator--() && noexcept;
// value_type operator--(int) && noexcept;

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, std::size_t>
struct CheckSimdReferenceUnaryOperators {
  template <class SimdAbi>
  void operator()() const {
    ex::simd<T, SimdAbi> origin_simd(static_cast<T>(3));
    static_assert(noexcept(++origin_simd[0]));
    assert(((T)(++origin_simd[0]) == static_cast<T>(4)) && ((T)origin_simd[0] == static_cast<T>(4)));
    static_assert(noexcept(origin_simd[0]++));
    assert(((T)(origin_simd[0]++) == static_cast<T>(4)) && ((T)origin_simd[0] == static_cast<T>(5)));
    static_assert(noexcept(--origin_simd[0]));
    assert(((T)(--origin_simd[0]) == static_cast<T>(4)) && ((T)origin_simd[0] == static_cast<T>(4)));
    static_assert(noexcept(origin_simd[0]--));
    assert(((T)(origin_simd[0]--) == static_cast<T>(4)) && ((T)origin_simd[0] == static_cast<T>(3)));
  }
};

template <class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_pre_increment : std::false_type {};

template <class T, class SimdAbi>
struct has_pre_increment<T, SimdAbi, std::void_t<decltype(++std::declval<ex::simd<T, SimdAbi>>()[0])>>
    : std::true_type {};

template <class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_post_increment : std::false_type {};

template <class T, class SimdAbi>
struct has_post_increment<T, SimdAbi, std::void_t<decltype(std::declval<ex::simd<T, SimdAbi>>()[0]++)>>
    : std::true_type {};

template <class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_pre_decrement : std::false_type {};

template <class T, class SimdAbi>
struct has_pre_decrement<T, SimdAbi, std::void_t<decltype(--std::declval<ex::simd<T, SimdAbi>>()[0])>>
    : std::true_type {};

template <class T, class SimdAbi = ex::simd_abi::compatible<T>, class = void>
struct has_post_decrement : std::false_type {};

template <class T, class SimdAbi>
struct has_post_decrement<T, SimdAbi, std::void_t<decltype(std::declval<ex::simd<T, SimdAbi>>()[0]--)>>
    : std::true_type {};

template <class T, std::size_t>
struct CheckSimdReferenceUnaryTraits {
  template <class SimdAbi>
  void operator()() {
    static_assert(has_pre_increment<T, SimdAbi>::value);
    static_assert(has_post_increment<T, SimdAbi>::value);
    static_assert(has_pre_decrement<T, SimdAbi>::value);
    static_assert(has_post_decrement<T, SimdAbi>::value);
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdReferenceUnaryOperators>();
  test_all_simd_abi<CheckSimdReferenceUnaryTraits>();
  return 0;
}
