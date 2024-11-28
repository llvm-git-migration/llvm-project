// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RADIX_SORT_H
#define _LIBCPP___ALGORITHM_RADIX_SORT_H

#include <__algorithm/copy.h>
#include <__algorithm/for_each.h>
#include <__bit/countl.h>
#include <__config>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/move_iterator.h>
#include <__iterator/next.h>
#include <__iterator/reverse_iterator.h>
#include <__numeric/partial_sum.h>
#include <__type_traits/decay.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_unsigned.h>
#include <__type_traits/make_unsigned.h>
#include <__utility/forward.h>
#include <__utility/integer_sequence.h>
#include <__utility/move.h>
#include <__utility/pair.h>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 14

inline _LIBCPP_HIDE_FROM_ABI void __variadic_expansion_dummy(initializer_list<int>) {}

#  define _EXPAND_VARIADIC(expression) std::__variadic_expansion_dummy({(expression, 0)...})

template <typename _Iterator>
_LIBCPP_HIDE_FROM_ABI constexpr auto __move_assign_please(_Iterator __i)
    -> enable_if_t<is_move_assignable<typename iterator_traits<_Iterator>::value_type>::value,
                   move_iterator<_Iterator> > {
  return std::make_move_iterator(std::move(__i));
}

template <typename _Iterator>
_LIBCPP_HIDE_FROM_ABI constexpr auto __move_assign_please(_Iterator __i)
    -> enable_if_t<not is_move_assignable<typename iterator_traits<_Iterator>::value_type>::value, _Iterator> {
  return __i;
}

template <typename _UnsignedInteger>
_LIBCPP_HIDE_FROM_ABI constexpr _UnsignedInteger __intlog2(_UnsignedInteger __n) {
  static_assert(is_integral<_UnsignedInteger>::value, "Must be an integral type");
  static_assert(is_unsigned<_UnsignedInteger>::value, "Must be unsigned");

  return numeric_limits<_UnsignedInteger>::digits - 1 - std::__countl_zero(__n);
}

template <typename _InputIterator, typename _OutputIterator>
_LIBCPP_HIDE_FROM_ABI pair<_OutputIterator, typename iterator_traits<_InputIterator>::value_type>
__partial_sum_max(_InputIterator __first, _InputIterator __last, _OutputIterator __result) {
  if (__first == __last)
    return {__result, 0};

  auto __max                                                 = *__first;
  typename iterator_traits<_InputIterator>::value_type __sum = *__first;
  *__result                                                  = __sum;

  while (++__first != __last) {
    if (__max < *__first) {
      __max = *__first;
    }
    __sum       = std::move(__sum) + *__first;
    *++__result = __sum;
  }
  return {++__result, __max};
}

template <typename _Value, typename _Map, typename _Radix>
struct __radix_sort_traits {
  using image_type = decay_t<invoke_result_t<_Map, _Value> >;
  static_assert(is_integral<image_type>::value, "");
  static_assert(is_unsigned<image_type>::value, "");

  using radix_type = decay_t<invoke_result_t<_Radix, image_type> >;
  static_assert(is_integral<radix_type>::value, "");

  constexpr static auto __radix_value_range = numeric_limits<radix_type>::max() + 1;
  constexpr static auto __radix_size        = std::__intlog2<uint64_t>(__radix_value_range);
  constexpr static auto __radix_count       = sizeof(image_type) * CHAR_BIT / __radix_size;
};

template <typename _Value, typename _Map>
struct __counting_sort_traits {
  using image_type = decay_t<invoke_result_t<_Map, _Value> >;
  static_assert(is_integral<image_type>::value, "");
  static_assert(is_unsigned<image_type>::value, "");

  constexpr static const auto __value_range = numeric_limits<image_type>::max() + 1;
  constexpr static auto __radix_size        = std::__intlog2<uint64_t>(__value_range);
};

template <typename _Radix>
_LIBCPP_HIDE_FROM_ABI auto __nth_radix(size_t __radix_number, _Radix __radix) {
  return [__radix_number, __radix = std::move(__radix)](auto __n) {
    using value_type = decltype(__n);
    static_assert(is_integral<value_type>::value, "");
    static_assert(is_unsigned<value_type>::value, "");
    using traits = __counting_sort_traits<value_type, _Radix>;

    return __radix(static_cast<value_type>(__n >> traits::__radix_size * __radix_number));
  };
}

template <typename _ForwardIterator, typename _Map, typename _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI void
__count(_ForwardIterator __first, _ForwardIterator __last, _Map __map, _RandomAccessIterator __counters) {
  std::for_each(__first, __last, [&__counters, &__map](const auto& __preimage) { ++__counters[__map(__preimage)]; });
}

template <typename _ForwardIterator, typename _Map, typename _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI void
__collect(_ForwardIterator __first, _ForwardIterator __last, _Map __map, _RandomAccessIterator __counters) {
  using value_type = typename iterator_traits<_ForwardIterator>::value_type;
  using traits     = __counting_sort_traits<value_type, _Map>;

  std::__count(__first, __last, __map, __counters);

  const auto __counters_end = __counters + traits::__value_range;
  std::partial_sum(__counters, __counters_end, __counters);
}

template <typename _ForwardIterator, typename _RandomAccessIterator1, typename _Map, typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI void
__dispose(_ForwardIterator __first,
          _ForwardIterator __last,
          _RandomAccessIterator1 __result,
          _Map __map,
          _RandomAccessIterator2 __counters) {
  std::for_each(__first, __last, [&__result, &__counters, &__map](auto&& __preimage) {
    auto __index      = __counters[__map(__preimage)]++;
    __result[__index] = std::forward<decltype(__preimage)>(__preimage);
  });
}

template <typename _BidirectionalIterator,
          typename _RandomAccessIterator1,
          typename _Map,
          typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI void dispose_backward(
    _BidirectionalIterator __first,
    _BidirectionalIterator __last,
    _RandomAccessIterator1 __result,
    _Map __map,
    _RandomAccessIterator2 __counters) {
  std::for_each(std::make_reverse_iterator(__last),
                std::make_reverse_iterator(__first),
                [&__result, &__counters, &__map](auto&& __preimage) {
                  auto __index      = --__counters[__map(__preimage)];
                  __result[__index] = std::forward<decltype(__preimage)>(__preimage);
                });
}

template <typename _ForwardIterator,
          typename _Map,
          typename _Radix,
          typename _RandomAccessIterator1,
          typename _RandomAccessIterator2,
          size_t... _Radices>
_LIBCPP_HIDE_FROM_ABI bool __collect_impl(
    _ForwardIterator __first,
    _ForwardIterator __last,
    _Map __map,
    _Radix __radix,
    _RandomAccessIterator1 __counters,
    _RandomAccessIterator2 __maximums,
    index_sequence<_Radices...>) {
  using value_type                   = typename iterator_traits<_ForwardIterator>::value_type;
  constexpr auto __radix_value_range = __radix_sort_traits<value_type, _Map, _Radix>::__radix_value_range;

  auto __previous  = numeric_limits<invoke_result_t<_Map, value_type> >::min();
  auto __is_sorted = true;
  std::for_each(__first, __last, [&__counters, &__map, &__radix, &__previous, &__is_sorted](const auto& value) {
    auto __current = __map(value);
    __is_sorted &= (__current >= __previous);
    __previous = __current;

    _EXPAND_VARIADIC(++__counters[_Radices][std::__nth_radix(_Radices, __radix)(__current)]);
  });

  _EXPAND_VARIADIC(
      __maximums[_Radices] =
          std::__partial_sum_max(__counters[_Radices], __counters[_Radices] + __radix_value_range, __counters[_Radices])
              .second);

  return __is_sorted;
}

template <typename _ForwardIterator,
          typename _Map,
          typename _Radix,
          typename _RandomAccessIterator1,
          typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI bool
__collect(_ForwardIterator __first,
          _ForwardIterator __last,
          _Map __map,
          _Radix __radix,
          _RandomAccessIterator1 __counters,
          _RandomAccessIterator2 __maximums) {
  using value_type             = typename iterator_traits<_ForwardIterator>::value_type;
  constexpr auto __radix_count = __radix_sort_traits<value_type, _Map, _Radix>::__radix_count;
  return std::__collect_impl(
      __first, __last, __map, __radix, __counters, __maximums, make_index_sequence<__radix_count>());
}

template <typename _BidirectionalIterator,
          typename _RandomAccessIterator1,
          typename _Map,
          typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI void __dispose_backward(
    _BidirectionalIterator __first,
    _BidirectionalIterator __last,
    _RandomAccessIterator1 __result,
    _Map __map,
    _RandomAccessIterator2 __counters) {
  std::for_each(std::make_reverse_iterator(__last),
                std::make_reverse_iterator(__first),
                [&__result, &__counters, &__map](auto&& preimage) {
                  auto __index      = --__counters[__map(preimage)];
                  __result[__index] = std::forward<decltype(preimage)>(preimage);
                });
}

template <typename _ForwardIterator, typename _RandomAccessIterator, typename _Map>
_LIBCPP_HIDE_FROM_ABI _RandomAccessIterator
__counting_sort_impl(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __result, _Map __map) {
  using value_type = typename iterator_traits<_ForwardIterator>::value_type;
  using traits     = __counting_sort_traits<value_type, _Map>;

  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  difference_type __counters[traits::__value_range + 1] = {0};

  std::__collect(__first, __last, __map, std::next(std::begin(__counters)));
  std::__dispose(__first, __last, __result, __map, std::begin(__counters));

  return __result + __counters[traits::__value_range];
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Map, typename _Radix>
_LIBCPP_HIDE_FROM_ABI typename enable_if<
    __radix_sort_traits<typename iterator_traits<_RandomAccessIterator1>::value_type, _Map, _Radix>::__radix_count == 1,
    void>::type
__radix_sort_impl(_RandomAccessIterator1 __first,
                  _RandomAccessIterator1 __last,
                  _RandomAccessIterator2 buffer,
                  _Map __map,
                  _Radix __radix) {
  auto __buffer_end = std::__counting_sort_impl(
      std::__move_assign_please(__first),
      std::__move_assign_please(__last),
      buffer,
      [&__map, &__radix](const auto& value) { return __radix(__map(value)); });

  std::copy(std::__move_assign_please(buffer), std::__move_assign_please(__buffer_end), __first);
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Map, typename _Radix>
_LIBCPP_HIDE_FROM_ABI typename enable_if<
    __radix_sort_traits<typename iterator_traits<_RandomAccessIterator1>::value_type, _Map, _Radix>::__radix_count %
            2 ==
        0,
    void>::type
__radix_sort_impl(_RandomAccessIterator1 __first,
                  _RandomAccessIterator1 __last,
                  _RandomAccessIterator2 __buffer_begin,
                  _Map __map,
                  _Radix __radix) {
  using value_type = typename iterator_traits<_RandomAccessIterator1>::value_type;
  using traits     = __radix_sort_traits<value_type, _Map, _Radix>;

  using difference_type = typename iterator_traits<_RandomAccessIterator1>::difference_type;
  difference_type __counters[traits::__radix_count][traits::__radix_value_range] = {{0}};
  difference_type __maximums[traits::__radix_count]                              = {0};
  const auto __is_sorted = std::__collect(__first, __last, __map, __radix, __counters, __maximums);
  if (not __is_sorted) {
    const auto __range_size = std::distance(__first, __last);
    auto __buffer_end       = __buffer_begin + __range_size;
    for (size_t __radix_number = 0; __radix_number < traits::__radix_count; __radix_number += 2) {
      const auto __n0th_is_single = __maximums[__radix_number] == __range_size;
      const auto __n1th_is_single = __maximums[__radix_number + 1] == __range_size;

      if (__n0th_is_single && __n1th_is_single) {
        continue;
      }

      if (__n0th_is_single) {
        std::copy(std::__move_assign_please(__first), std::__move_assign_please(__last), __buffer_begin);
      } else {
        auto __n0th = [__radix_number, &__map, &__radix](const auto& __v) {
          return std::__nth_radix(__radix_number, __radix)(__map(__v));
        };
        std::__dispose_backward(
            std::__move_assign_please(__first),
            std::__move_assign_please(__last),
            __buffer_begin,
            __n0th,
            __counters[__radix_number]);
      }

      if (__n1th_is_single) {
        std::copy(std::__move_assign_please(__buffer_begin), std::__move_assign_please(__buffer_end), __first);
      } else {
        auto __n1th = [__radix_number, &__map, &__radix](const auto& __v) {
          return std::__nth_radix(__radix_number + 1, __radix)(__map(__v));
        };
        std::__dispose_backward(
            std::__move_assign_please(__buffer_begin),
            std::__move_assign_please(__buffer_end),
            __first,
            __n1th,
            __counters[__radix_number + 1]);
      }
    }
  }
}

_LIBCPP_HIDE_FROM_ABI constexpr auto __to_unsigned(bool __b) { return __b; }

template <typename _Ip>
_LIBCPP_HIDE_FROM_ABI constexpr auto __to_unsigned(_Ip __n) {
  constexpr const auto __min_value = numeric_limits<_Ip>::min();
  return static_cast<make_unsigned_t<_Ip> >(__n ^ __min_value);
}

struct __identity_fn {
  template <typename _Tp>
  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator()(_Tp&& __value) const {
    return std::forward<_Tp>(__value);
  }
};

struct __low_byte_fn {
  template <typename _Ip>
  _LIBCPP_HIDE_FROM_ABI constexpr uint8_t operator()(_Ip __integer) const {
    static_assert(is_integral<_Ip>::value, "");
    static_assert(is_unsigned<_Ip>::value, "");

    return static_cast<uint8_t>(__integer & 0xff);
  }
};

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Map, typename _Radix>
_LIBCPP_HIDE_FROM_ABI void
__radix_sort(_RandomAccessIterator1 __first,
             _RandomAccessIterator1 __last,
             _RandomAccessIterator2 buffer,
             _Map __map,
             _Radix __radix) {
  auto __map_to_unsigned = [__map = std::move(__map)](const auto& x) { return std::__to_unsigned(__map(x)); };
  std::__radix_sort_impl(__first, __last, buffer, __map_to_unsigned, __radix);
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI void
__radix_sort(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 buffer) {
  std::__radix_sort(__first, __last, buffer, __identity_fn{}, __low_byte_fn{});
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI bool __radix_sort(
    _RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 buffer, _BoolConstant<true>) {
  std::__radix_sort(__first, __last, buffer, __identity_fn{}, __low_byte_fn{});
  return true;
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
_LIBCPP_HIDE_FROM_ABI bool
__radix_sort(_RandomAccessIterator1, _RandomAccessIterator1, _RandomAccessIterator2, _BoolConstant<false>) {
  return false;
}

#  undef _EXPAND_VARIADIC

#else // _LIBCPP_STD_VER > 14

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, bool _EnableRadixSort>
_LIBCPP_HIDE_FROM_ABI bool
__radix_sort(_RandomAccessIterator1, _RandomAccessIterator1, _RandomAccessIterator2, _BoolConstant<_EnableRadixSort>) {
  return false;
}

#endif // _LIBCPP_STD_VER > 14

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RADIX_SORT_H
