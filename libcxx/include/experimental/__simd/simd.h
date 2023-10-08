// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H
#define _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H

#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <cstddef>
#include <experimental/__config>
#include <experimental/__simd/avx512.h>
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/reference.h>
#include <experimental/__simd/traits.h>
#include <experimental/__simd/utility.h>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

// class template simd [simd.class]
// TODO: implement simd class
template <class _Tp, class _Abi>
class simd {
  using _Impl    = __simd_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_;

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Tp, _Storage, value_type>;
  using mask_type  = simd_mask<_Tp, _Abi>;
  using abi_type   = _Abi;

  static _LIBCPP_HIDE_FROM_ABI constexpr size_t size() noexcept { return simd_size_v<value_type, abi_type>; }

  _LIBCPP_HIDE_FROM_ABI simd() noexcept = default;

  template <class _Up, class _Flags>
  _LIBCPP_HIDE_FROM_ABI simd(const _Up* __data, _Flags) noexcept : __s_(_Impl::__load(__data)) {}

  // broadcast constructor
  template <class _Up, enable_if_t<__can_broadcast_v<value_type, __remove_cvref_t<_Up>>, int> = 0>
  _LIBCPP_HIDE_FROM_ABI simd(_Up&& __v) noexcept : __s_(_Impl::__broadcast(static_cast<value_type>(__v))) {}

  // implicit type conversion constructor
  template <class _Up,
            enable_if_t<!is_same_v<_Up, _Tp> && is_same_v<abi_type, simd_abi::fixed_size<size()>> &&
                            __is_non_narrowing_convertible_v<_Up, value_type>,
                        int> = 0>
  _LIBCPP_HIDE_FROM_ABI simd(const simd<_Up, simd_abi::fixed_size<size()>>& __v) noexcept {
    for (size_t __i = 0; __i < size(); __i++) {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  // generator constructor
  template <class _Generator, enable_if_t<__can_generate_v<value_type, _Generator, size()>, int> = 0>
  explicit _LIBCPP_HIDE_FROM_ABI simd(_Generator&& __g) noexcept
      : __s_(_Impl::__generate(std::forward<_Generator>(__g))) {}

  _LIBCPP_HIDE_FROM_ABI simd(__from_storage_t, _Storage __data) noexcept : __s_(__data) {}

  // scalar access [simd.subscr]
  _LIBCPP_HIDE_FROM_ABI reference operator[](size_t __i) noexcept { return reference(__s_, __i); }
  _LIBCPP_HIDE_FROM_ABI value_type operator[](size_t __i) const noexcept { return __s_.__get(__i); }

  _LIBCPP_HIDE_FROM_ABI _Storage __get_data() const { return __s_; }

#  ifdef __AVX512F__
  template <int __comparator>
  static _LIBCPP_HIDE_FROM_ABI auto __cmp(_Storage __lhs_wrapped, _Storage __rhs_wrapped) {
      auto __lhs = __lhs_wrapped.__data;
      auto __rhs = __rhs_wrapped.__data;
      constexpr auto __element_size  = sizeof(_Tp);
      constexpr auto __element_count = size();
      if constexpr (__element_size == 1) {
        if constexpr (__element_count == 16) {
          return _mm_cmp_epi8_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 32) {
          return _mm256_cmp_epi8_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 64) {
          return _mm512_cmp_epi8_mask(__lhs, __rhs, __comparator);
        } else {
          static_assert(__element_count == 0, "Unexpected size");
        }
      } else if constexpr (__element_size == 2) {
        if constexpr (__element_count == 8) {
          return _mm_cmp_epi16_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 16) {
          return _mm256_cmp_epi16_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 32) {
          return _mm512_cmp_epi16_mask(__lhs, __rhs, __comparator);
        } else {
          static_assert(__element_count == 0, "Unexpected size");
        }
      } else if constexpr (__element_size == 4) {
        if constexpr (__element_count == 4) {
          return _mm_cmp_epi32_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 8) {
          return _mm256_cmp_epi32_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 16) {
          return _mm512_cmp_epi32_mask(__lhs, __rhs, __comparator);
        } else {
          static_assert(__element_count == 0, "Unexpected size");
        }
      } else if constexpr (__element_size == 8) {
        if constexpr (__element_count == 2) {
          return _mm_cmp_epi64_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 4) {
          return _mm256_cmp_epi64_mask(__lhs, __rhs, __comparator);
        } else if constexpr (__element_count == 8) {
          return _mm512_cmp_epi64_mask(__lhs, __rhs, __comparator);
        } else {
          static_assert(__element_count == 0, "Unexpected size");
        }
      }
  }
#  endif

  friend _LIBCPP_HIDE_FROM_ABI mask_type operator==(const simd& __lhs, const simd& __rhs) noexcept {
#ifdef __AVX512F__
    if constexpr (simd_abi::__is_avx512_v<_Abi>) {
      return {__from_storage, {__cmp<_MM_CMPINT_EQ>(__lhs.__s_, __rhs.__s_)}};
    } else
#endif
    {
      mask_type __result;
      for (int __i = 0; __i != size(); ++__i)
        __result[__i] = __lhs[__i] == __rhs[__i];
      return __result;
    }
  }
};

template <class _Tp, class _Abi>
inline constexpr bool is_simd_v<simd<_Tp, _Abi>> = true;

template <class _Tp>
using native_simd = simd<_Tp, simd_abi::native<_Tp>>;

template <class _Tp, int _Np>
using fixed_size_simd = simd<_Tp, simd_abi::fixed_size<_Np>>;

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#endif // _LIBCPP_EXPERIMENTAL___SIMD_SIMD_H
