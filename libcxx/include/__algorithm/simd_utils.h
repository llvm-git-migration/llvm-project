//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SIMD_UTILS_H
#define _LIBCPP___ALGORITHM_SIMD_UTILS_H

#include <__bit/bit_cast.h>
#include <__bit/countr.h>
#include <__config>
#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_same.h>
#include <__utility/integer_sequence.h>
#include <cstddef>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 14 && defined(_LIBCPP_COMPILER_CLANG_BASED)
#  define _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS 1
#else
#  define _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS 0
#endif

#if _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS && !defined(__OPTIMIZE_SIZE__)
#  define _LIBCPP_VECTORIZE_ALGORITHMS 1
#else
#  define _LIBCPP_VECTORIZE_ALGORITHMS 0
#endif

#if _LIBCPP_HAS_ALGORITHM_VECTOR_UTILS

_LIBCPP_BEGIN_NAMESPACE_STD

// This isn't specialized for 64 byte vectors on purpose. They have the potential to significantly reduce performance
// in mixed simd/non-simd workloads and don't provide any performance improvement for currently vectorized algorithms
// as far as benchmarks are concerned.
#  if defined(__AVX__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 32 / sizeof(_Tp);
#  elif defined(__SSE__) || defined(__ARM_NEON__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 16 / sizeof(_Tp);
#  elif defined(__MMX__)
template <class _Tp>
inline constexpr size_t __native_vector_size = 8 / sizeof(_Tp);
#  else
template <class _Tp>
inline constexpr size_t __native_vector_size = 1;
#  endif

template <class _ArithmeticT, size_t _Np>
using __simd_vector __attribute__((__ext_vector_type__(_Np))) = _ArithmeticT;

template <class _VecT>
inline constexpr size_t __simd_vector_size_v = []<bool _False = false>() -> size_t {
  static_assert(_False, "Not a vector!");
}();

template <class _Tp, size_t _Np>
inline constexpr size_t __simd_vector_size_v<__simd_vector<_Tp, _Np>> = _Np;

template <class _Tp, size_t _Np>
_LIBCPP_HIDE_FROM_ABI _Tp __simd_vector_underlying_type_impl(__simd_vector<_Tp, _Np>) {
  return _Tp{};
}

template <class _VecT>
using __simd_vector_underlying_type_t = decltype(std::__simd_vector_underlying_type_impl(_VecT{}));

// This isn't inlined without always_inline when loading chars.
template <class _VecT, class _Tp>
_LIBCPP_NODISCARD _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _VecT __load_vector(const _Tp* __ptr) noexcept {
  return [=]<size_t... _Indices>(index_sequence<_Indices...>) _LIBCPP_ALWAYS_INLINE noexcept {
    return _VecT{__ptr[_Indices]...};
  }(make_index_sequence<__simd_vector_size_v<_VecT>>{});
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI bool __all_of(__simd_vector<_Tp, _Np> __vec) noexcept {
  return __builtin_reduce_and(__builtin_convertvector(__vec, __simd_vector<bool, _Np>));
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI size_t __find_first_set(__simd_vector<_Tp, _Np> __vec) noexcept {
  using __mask_vec = __simd_vector<bool, _Np>;

  auto __impl = [&]<class _MaskT>(_MaskT) noexcept {
    return std::__countr_zero(std::__bit_cast<_MaskT>(__builtin_convertvector(__vec, __mask_vec)));
  };

  if constexpr (sizeof(__mask_vec) == sizeof(uint8_t)) {
    return __impl(uint8_t{});
  } else if constexpr (sizeof(__mask_vec) == sizeof(uint16_t)) {
    return __impl(uint16_t{});
  } else if constexpr (sizeof(__mask_vec) == sizeof(uint32_t)) {
    return __impl(uint32_t{});
  } else if constexpr (sizeof(__mask_vec) == sizeof(uint64_t)) {
    return __impl(uint64_t{});
  } else {
    static_assert(sizeof(__mask_vec) == 0, "unexpected required size for mask integer type");
    return 0;
  }
}

template <class _Tp, size_t _Np>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI size_t __find_first_not_set(__simd_vector<_Tp, _Np> __vec) noexcept {
  return std::__find_first_set(~__vec);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 14 && __has_attribute(__ext_vector_type__) && __has_builtin(__builtin_reduce_and) &&
       // __has_builtin(__builtin_convertvector)

#endif // _LIBCPP___ALGORITHM_SIMD_UTILS_H
