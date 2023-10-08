//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_AVX512_H
#define _LIBCPP_EXPERIMENTAL___SIMD_AVX512_H

#include <__bit/bit_ceil.h>
#include <experimental/__config>
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/vec_ext.h>

#if __has_include(<immintrin.h>)
#  include <immintrin.h>
#endif

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL) && defined(__AVX512F__)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {
namespace simd_abi {
template <int _Np>
struct __avx512 {
  static constexpr size_t __simd_size = _Np;
};

template <class _Tp>
inline constexpr bool __is_avx512_v = false;

template <int _Np>
inline constexpr bool __is_avx512_v<__avx512<_Np>> = true;
} // namespace simd_abi

template <int _Np>
inline constexpr bool is_abi_tag_v<simd_abi::__avx512<_Np>> = _Np > 0 && _Np <= 64;

template <class _Tp, int _Np>
struct __simd_storage<_Tp, simd_abi::__avx512<_Np>> : __simd_storage<_Tp, simd_abi::__vec_ext<_Np>> {};

template <class _Tp, int _Np>
struct __mask_storage<_Tp, simd_abi::__avx512<_Np>> {
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __get_mask_t() {
    if constexpr (_Np <= 8)
      return __mmask8{};
    else if constexpr (_Np <= 16)
      return __mmask16{};
    else if constexpr (_Np <= 32)
      return __mmask32{};
    else if constexpr (_Np <= 64)
      return __mmask64{};
    else
      static_assert(_Np == -1, "Unexpected size");
  }
  decltype(__get_mask_t()) __mask_;

  _LIBCPP_HIDE_FROM_ABI bool __get(size_t __index) const noexcept { return __mask_ & 1 << __index; }
  _LIBCPP_HIDE_FROM_ABI void __set(size_t __index, bool __value) noexcept {
    if (__value)
      __mask_ |= 1 << __index;
    else
      __mask_ &= ~(1 << __index);
  }
};

template <class _Tp, int _Np>
struct __simd_operations<_Tp, simd_abi::__avx512<_Np>> : __simd_operations<_Tp, simd_abi::__vec_ext<_Np>> {};

template <class _Tp, int _Np>
struct __mask_operations<_Tp, simd_abi::__avx512<_Np>> {
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__avx512<_Np>>;

  _LIBCPP_HIDE_FROM_ABI static _MaskStorage __broadcast(bool __v) noexcept {
    if (__v)
      return {numeric_limits<_MaskStorage>::max()};
    else
      return {0};
  }

  _LIBCPP_HIDE_FROM_ABI static bool all_of(_MaskStorage __mask) noexcept {
    return __mask.__mask_ == __broadcast(true).__mask_;
  }
};
} // namespace parallelism_v2

_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL) && defined(__AVX512F__)

#endif // _LIBCPP_EXPERIMENTAL___SIMD_AVX512_H
