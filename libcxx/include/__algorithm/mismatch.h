// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MISMATCH_H
#define _LIBCPP___ALGORITHM_MISMATCH_H

#include <__algorithm/comp.h>
#include <__algorithm/unwrap_iter.h>
#include <__algorithm/vectorization.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_equality_comparable.h>
#include <__utility/align_down.h>
#include <__utility/move.h>
#include <__utility/pair.h>
#include <experimental/__simd/feature_traits.h>
#include <experimental/__simd/simd.h>
#include <experimental/__simd/simd_mask.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InIter1, class _Sent1, class _InIter2, class _Pred, class _Proj1, class _Proj2>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter1, _InIter2>
__mismatch_loop(_InIter1 __first1, _Sent1 __last1, _InIter2 __first2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2) {
  while (__first1 != __last1) {
    if (!std::__invoke(__pred, std::__invoke(__proj1, *__first1), std::__invoke(__proj2, *__first2)))
      break;
    ++__first1;
    ++__first2;
  }
  return {std::move(__first1), std::move(__first2)};
}

#if _LIBCPP_CAN_VECTORIZE_ALGORIHTMS
template <class _Tp>
struct __mismatch_vector_impl {
  template <bool _VectorizeFloatingPoint>
  static constexpr bool __can_vectorize =
      (__libcpp_is_trivially_equality_comparable<_Tp, _Tp>::value && __fits_in_vector<_Tp> &&
       alignof(_Tp) >= alignof(__get_arithmetic_type<_Tp>)) ||
      (_VectorizeFloatingPoint && is_floating_point_v<_Tp>);

  using __vec         = __arithmetic_vec<_Tp>;
  using __mask_traits = experimental::__mask_traits<typename __vec::value_type, typename __vec::abi_type>;
  static constexpr size_t __unroll_count = 4;

  struct __result {
    _Tp* __iter1;
    _Tp* __iter2;
    bool __matched;
  };

  _LIBCPP_HIDE_FROM_ABI static __result __prologue(_Tp* __first1, _Tp* __last1, _Tp* __first2) {
    if constexpr (__mask_traits::__has_maskload) {
      auto __first_aligned = std::__align_down(__vec::size(), __first1);
      auto __offset        = __first1 - __first_aligned;
      auto __checked_size  = __vec::size() - __offset;
      if (__checked_size < __last1 - __first1)
        return {__first1, __first2, false};
      auto __second_aligned = __first2 - __offset;
      auto __mask           = __mask_traits::__mask_with_last_enabled(__checked_size);
      __vec __lhs =
          __mask_traits::__maskload_unaligned(reinterpret_cast<typename __vec::value_type*>(__first_aligned), __mask);
      __vec __rhs =
          __mask_traits::__maskload_unaligned(reinterpret_cast<typename __vec::value_type*>(__second_aligned), __mask);
      auto __res      = __mask_traits::__mask_cmp_eq(__mask, __lhs, __rhs);
      auto __inv_mask = ~__mask.__get_data().__mask_;
      if ((__res.__get_data().__mask_ & __mask.__get_data().__mask_) != __mask.__get_data().__mask_) {
        auto __match_offset = experimental::find_first_set(decltype(__mask){
            experimental::__from_storage, {decltype(__res.__get_data().__mask_)(~__res.__get_data().__mask_)}});
        return {__first_aligned + __match_offset, __second_aligned + __match_offset, true};
      }
      return {__first_aligned + __vec::size(), __second_aligned + __vec::size(), false};
    } else {
      return {__first1, __first2, false};
    }
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE static __result __loop(_Tp* __first1, _Tp* __last1, _Tp* __first2) {
    while (__last1 - __first1 >= __unroll_count * __vec::size()) {
      __vec __lhs[__unroll_count];
      __vec __rhs[__unroll_count];

      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        __lhs[__i] = std::__load_as_arithmetic(__first1 + __i * __vec::size());
        __rhs[__i] = std::__load_as_arithmetic(__first2 + __i * __vec::size());
      }

      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        if (auto __res = __lhs[__i] == __rhs[__i]; !experimental::all_of(__res)) {
          auto __offset = __i * __vec::size() + experimental::find_first_set(__res);
          return {__first1 + __offset, __first2 + __offset, true};
        }
      }

      __first1 += __unroll_count * __vec::size();
      __first2 += __unroll_count * __vec::size();
    }
    return {__first1, __first2, __first1 == __last1};
  }

  _LIBCPP_HIDE_FROM_ABI static pair<_Tp*, _Tp*> __epilogue(_Tp* __first1, _Tp* __last1, _Tp* __first2) {
    if constexpr (__mask_traits::__has_maskload) {
      auto __size = __last1 - __first1;
      auto __mask = __mask_traits::__mask_with_first_enabled(__size);
      __vec __lhs =
          __mask_traits::__maskload_unaligned(reinterpret_cast<typename __vec::value_type*>(__first1), __mask);
      __vec __rhs =
          __mask_traits::__maskload_unaligned(reinterpret_cast<typename __vec::value_type*>(__first2), __mask);
      auto __res      = __mask_traits::__mask_cmp_eq(__mask, __lhs, __rhs);
      auto __inv_mask = ~__mask.__get_data().__mask_;
      if ((__res.__get_data().__mask_ | __inv_mask) != decltype(__mask){true}.__get_data().__mask_) {
        auto __offset = experimental::find_first_set(__res);
        return {__first1 + __offset, __first2 + __offset};
      }
      return {__first1 + __size, __first2 + __size};
    } else {
      return std::__mismatch_loop(__first1, __last1, __first2, __equal_to(), __identity(), __identity());
    }
  }
};
#endif // _LIBCPP_CAN_VECTORIZE_ALGORIHTMS

template <class _InIter1, class _Sent1, class _InIter2, class _Pred, class _Proj1, class _Proj2>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter1, _InIter2>
__mismatch(_InIter1 __first1, _Sent1 __last1, _InIter2 __first2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2) {
  return std::__mismatch_loop(__first1, __last1, __first2, __pred, __proj1, __proj2);
}

#if _LIBCPP_VECTORIZE_CLASSIC_ALGORITHMS
template <
    class _Tp,
    class _Pred,
    class _Proj1,
    class _Proj2,
    enable_if_t<
        __desugars_to<__equal_tag, _Pred, _Tp, _Tp>::value && __is_identity<_Proj1>::value &&
            __is_identity<_Proj2>::value &&
            __mismatch_vector_impl<_Tp>::template __can_vectorize<_LIBCPP_VECTORIZE_FLOATING_POINT_CLASSIC_ALGORITHMS>,
        int> = 0>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI inline constexpr pair<_Tp*, _Tp*>
__mismatch(_Tp* __first1, _Tp* __last1, _Tp* __first2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2) {
  if (__libcpp_is_constant_evaluated())
    return std::__mismatch_loop(__first1, __last1, __first2, __pred, __proj1, __proj2);

  using __impl = __mismatch_vector_impl<_Tp>;

  // auto [__piter1, __piter2, __pmatch] = __impl::__prologue(__first1, __last1, __first2);
  // if (__pmatch)
  //   return {__piter1, __piter2};

  auto [__iter1, __iter2, __matched] = __impl::__loop(__first1, __last1, __first2);
  if (__matched)
    return {__iter1, __iter2};

  return __impl::__epilogue(__first1, __last1, __first2);
}
#endif // _LIBCPP_VECTORIZE_ALGORITHMS

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred) {
  __identity __proj;
  auto __res = std::__mismatch(
      std::__unwrap_iter(__first1), std::__unwrap_iter(__last1), std::__unwrap_iter(__first2), __pred, __proj, __proj);
  return std::make_pair(std::__rewrap_iter(__first1, __res.first), std::__rewrap_iter(__first2, __res.second));
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2) {
  return std::mismatch(__first1, __last1, __first2, __equal_to());
}

#if _LIBCPP_STD_VER >= 14
template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1,
         _InputIterator1 __last1,
         _InputIterator2 __first2,
         _InputIterator2 __last2,
         _BinaryPredicate __pred) {
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      break;
  return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return std::mismatch(__first1, __last1, __first2, __last2, __equal_to());
}
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_MISMATCH_H
