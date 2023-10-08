//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_FEATURE_TRAITS_H
#define _LIBCPP_EXPERIMENTAL___SIMD_FEATURE_TRAITS_H

#include <__bit/has_single_bit.h>
#include <__config>
#include <__memory/assume_aligned.h>
#include <experimental/__simd/declaration.h>
#include <experimental/__simd/vec_ext.h>

#ifdef __AVX512F__
#  include <immintrin.h>
#endif

// The intrinsics cannot be portably qualified. This isn't super problematic, since we're only dealing with builtin
// types anyways.
// NOLINTBEGIN(libcpp-robust-against-adl)

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {

template <class _Tp, class _Abi, class = void>
struct __mask_traits {
  static constexpr bool __has_maskload  = false;
  static constexpr bool __has_maskstore = false;
};

template <uint64_t __base_pattern>
_LIBCPP_HIDE_FROM_ABI uint64_t __set_least_significant_bits(size_t __count) noexcept {
  uint64_t __bits = __base_pattern;
  __bits >>= 64 - __count;
  return __bits;
}

template <uint64_t __base_pattern>
_LIBCPP_HIDE_FROM_ABI uint64_t __set_most_significant_bits(size_t __count) noexcept {
  uint64_t __bits = __base_pattern;
  __bits <<= 64 - __count;
  __bits &= __base_pattern;
  return __bits;
}

#  ifdef __AVX512F__

template <class _Tp, size_t _Np>
struct __mask_traits<_Tp, simd_abi::__avx512<_Np>, enable_if_t<is_integral_v<_Tp>>> {
private:
  static constexpr size_t __element_count = _Np;
  static constexpr size_t __element_size  = sizeof(_Tp);

  using __simd_t = simd<_Tp, simd_abi::__avx512<_Np>>;
  using __mask_t = simd_mask<_Tp, simd_abi::__avx512<_Np>>;

  using __storage_t [[__gnu__::__vector_size__(_Np * sizeof(_Tp))]] = _Tp;

public:
#    ifdef __AVX512VL__
  static constexpr bool __has_maskload  = std::__has_single_bit(_Np);
  static constexpr bool __has_maskstore = __has_maskload;

  static _LIBCPP_HIDE_FROM_ABI __simd_t __maskload_unaligned(const _Tp* __ptr, __mask_t __mask_wrapped) {
    if constexpr (!__has_maskload) {
      return {};
    } else {
      __storage_t __data = [&] {
        auto __mask = __mask_wrapped.__get_data().__mask_;

        if constexpr (__element_size == 1) {
          if constexpr (__element_count == 16) {
            return _mm_maskz_loadu_epi8(__mask, __ptr);
          } else if constexpr (__element_count == 32) {
            return _mm256_maskz_loadu_epi8(__mask, __ptr);
          } else if constexpr (__element_count == 64) {
            return _mm512_maskz_loadu_epi8(__mask, __ptr);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 2) {
          if constexpr (__element_count == 8) {
            return _mm_maskz_loadu_epi16(__mask, __ptr);
          } else if constexpr (__element_count == 16) {
            return _mm256_maskz_loadu_epi16(__mask, __ptr);
          } else if constexpr (__element_count == 32) {
            return _mm512_maskz_loadu_epi16(__mask, __ptr);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 4) {
          if constexpr (__element_count == 4) {
            return _mm_maskz_loadu_epi32(__mask, __ptr);
          } else if constexpr (__element_count == 8) {
            return _mm256_maskz_loadu_epi32(__mask, __ptr);
          } else if constexpr (__element_count == 16) {
            return _mm512_maskz_loadu_epi32(__mask, __ptr);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 8) {
          if constexpr (__element_count == 2) {
            return _mm_maskz_loadu_epi64(__mask, __ptr);
          } else if constexpr (__element_count == 4) {
            return _mm256_maskz_loadu_epi64(__mask, __ptr);
          } else if constexpr (__element_count == 8) {
            return _mm512_maskz_loadu_epi64(__mask, __ptr);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else {
          static_assert(_Np == 3, "Unexpected size");
        }
      }();
      return {__from_storage, { __data }};
    }
  }

  _LIBCPP_HIDE_FROM_ABI void __maskstore(const _Tp* __ptr_raw, __simd_t __data_wrapped, __mask_t __mask_wrapped) {
    if constexpr (!__has_maskstore) {
      return;
    } else {
      [&] {
        auto __mask = __mask_wrapped.__get_data();
        auto __data = __data_wrapped.__get_data();
        auto __ptr  = std::__assume_aligned<sizeof(__storage_t)>(__ptr_raw);

        if constexpr (__element_size == 1) {
          if constexpr (__element_count == 16) {
            return _mm_mask_storeu_epi8(__ptr, __mask, __data);
          } else if constexpr (__element_count == 32) {
            return _mm256_mask_storeu_epi8(__ptr, __mask, __data);
          } else if constexpr (__element_count == 64) {
            return _mm512_mask_storeu_epi8(__ptr, __mask, __data);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 2) {
          if constexpr (__element_count == 8) {
            return _mm_mask_storeu_epi16(__ptr, __mask, __data);
          } else if constexpr (__element_count == 16) {
            return _mm256_mask_storeu_epi16(__ptr, __mask, __data);
          } else if constexpr (__element_count == 32) {
            return _mm512_mask_storeu_epi16(__ptr, __mask, __data);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 4) {
          if constexpr (__element_count == 4) {
            return _mm_mask_store_epi32(__ptr, __mask, __data);
          } else if constexpr (__element_count == 8) {
            return _mm256_mask_store_epi32(__ptr, __mask, __data);
          } else if constexpr (__element_count == 16) {
            return _mm512_mask_store_epi32(__ptr, __mask, __data);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 8) {
          if constexpr (__element_count == 2) {
            return _mm_mask_store_epi64(__ptr, __mask, __data);
          } else if constexpr (__element_count == 4) {
            return _mm256_mask_store_epi64(__ptr, __mask, __data);
          } else if constexpr (__element_count == 8) {
            return _mm512_mask_store_epi64(__ptr, __mask, __data);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        }
      }();
    }
  }

  static __mask_t __mask_with_first_enabled(size_t __n) noexcept {
    if constexpr (__element_count == 2) {
      auto __bitmask = experimental::__set_most_significant_bits<0x0000000000000003>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 4) {
      auto __bitmask = experimental::__set_most_significant_bits<0x000000000000000F>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 8) {
      auto __bitmask = experimental::__set_most_significant_bits<0x00000000000000FF>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 16) {
      auto __bitmask = experimental::__set_most_significant_bits<0x000000000000FFFF>(__n);
      return {__from_storage, { static_cast<__mmask16>(__bitmask) }};
    } else if constexpr (__element_count == 32) {
      auto __bitmask = experimental::__set_most_significant_bits<0x00000000FFFFFFFF>(__n);
      return {__from_storage, { static_cast<__mmask32>(__bitmask) }};
    } else if constexpr (__element_count == 64) {
      auto __bitmask = experimental::__set_most_significant_bits<0xFFFFFFFFFFFFFFFF>(__n);
      return {__from_storage, { static_cast<__mmask64>(__bitmask) }};
    }
  }

  static __mask_t __mask_with_last_enabled(size_t __n) noexcept {
    if constexpr (__element_count == 2) {
      auto __bitmask = experimental::__set_least_significant_bits<0x0000000000000003>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 4) {
      auto __bitmask = experimental::__set_least_significant_bits<0x000000000000000F>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 8) {
      auto __bitmask = experimental::__set_least_significant_bits<0x00000000000000FF>(__n);
      return {__from_storage, { static_cast<__mmask8>(__bitmask) }};
    } else if constexpr (__element_count == 16) {
      auto __bitmask = experimental::__set_least_significant_bits<0x000000000000FFFF>(__n);
      return {__from_storage, { static_cast<__mmask16>(__bitmask) }};
    } else if constexpr (__element_count == 32) {
      auto __bitmask = experimental::__set_least_significant_bits<0x00000000FFFFFFFF>(__n);
      return {__from_storage, { static_cast<__mmask32>(__bitmask) }};
    } else if constexpr (__element_count == 64) {
      auto __bitmask = experimental::__set_least_significant_bits<0xFFFFFFFFFFFFFFFF>(__n);
      return {__from_storage, { static_cast<__mmask64>(__bitmask) }};
    }
  }

  template <int __comparator>
  static _LIBCPP_HIDE_FROM_ABI __mask_t
  __mask_cmp_mask(__mask_t __mask_wrapped, __simd_t __lhs_wrapped, __simd_t __rhs_wrapped) {
    if constexpr (!__has_maskstore) {
      return;
    } else {
      auto __ret = [&] {
        auto __mask = __mask_wrapped.__get_data().__mask_;
        auto __lhs  = __lhs_wrapped.__get_data().__data;
        auto __rhs  = __rhs_wrapped.__get_data().__data;

        if constexpr (__element_size == 1) {
          if constexpr (__element_count == 16) {
            return _mm_mask_cmp_epi8_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 32) {
            return _mm256_mask_cmp_epi8_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 64) {
            return _mm512_mask_cmp_epi8_mask(__mask, __lhs, __rhs, __comparator);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 2) {
          if constexpr (__element_count == 8) {
            return _mm_mask_cmp_epi16_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 16) {
            return _mm256_mask_cmp_epi16_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 32) {
            return _mm512_mask_cmp_epi16_mask(__mask, __lhs, __rhs, __comparator);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 4) {
          if constexpr (__element_count == 4) {
            return _mm_mask_cmp_epi32_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 8) {
            return _mm256_mask_cmp_epi32_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 16) {
            return _mm512_mask_cmp_epi32_mask(__mask, __lhs, __rhs, __comparator);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        } else if constexpr (__element_size == 8) {
          if constexpr (__element_count == 2) {
            return _mm_mask_cmp_epi64_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 4) {
            return _mm256_mask_cmp_epi64_mask(__mask, __lhs, __rhs, __comparator);
          } else if constexpr (__element_count == 8) {
            return _mm512_mask_cmp_epi64_mask(__mask, __lhs, __rhs, __comparator);
          } else {
            static_assert(_Np == 3, "Unexpected size");
          }
        }
      }();
      return {__from_storage, {__ret}};
    }
  }

  static _LIBCPP_HIDE_FROM_ABI __mask_t __mask_cmp_eq(__mask_t __mask, __simd_t __lhs, __simd_t __rhs) noexcept {
    return __mask_cmp_mask<_MM_CMPINT_EQ>(__mask, __lhs, __rhs);
  }
#    else
  static constexpr bool __has_maskload  = false;
  static constexpr bool __has_maskstore = false;
#    endif
};

#  endif // __AVX512F__

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif

// NOLINTEND(libcpp-robust-against-adl)

#endif // _LIBCPP_EXPERIMENTAL___SIMD_FEATURE_TRAITS_H
