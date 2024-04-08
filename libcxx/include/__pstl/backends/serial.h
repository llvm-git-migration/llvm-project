//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_SERIAL_H
#define _LIBCPP___PSTL_BACKENDS_SERIAL_H

#include <__algorithm/find_if.h>
#include <__algorithm/for_each.h>
#include <__algorithm/merge.h>
#include <__algorithm/stable_sort.h>
#include <__algorithm/transform.h>
#include <__config>
#include <__numeric/transform_reduce.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/defaults.h>
#include <__utility/empty.h>
#include <__utility/forward.h>
#include <__utility/move.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

struct __serial_backend_tag {};

// Mandatory implementations of the computational basis
template <class _ExecutionPolicy>
struct __find_if<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Pred>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Pred&& __pred) const noexcept {
    return std::find_if(std::move(__first), std::move(__last), std::forward<_Pred>(__pred));
  }
};

template <class _ExecutionPolicy>
struct __for_each<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Function>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Function&& __func) const noexcept {
    std::for_each(std::move(__first), std::move(__last), std::forward<_Function>(__func));
    return __empty{};
  }
};

template <class _ExecutionPolicy>
struct __merge<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardOutIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _ForwardOutIterator __out,
      _Comp&& __comp) const noexcept {
    return std::merge(
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        std::move(__last2),
        std::move(__out),
        std::forward<_Comp>(__comp));
  }
};

template <class _ExecutionPolicy>
struct __stable_sort<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _RandomAccessIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp&& __comp) const noexcept {
    std::stable_sort(std::move(__first), std::move(__last), std::forward<_Comp>(__comp));
  }
};

template <class _ExecutionPolicy>
struct __transform<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __out, _UnaryOperation&& __op)
      const noexcept {
    return std::transform(std::move(__first), std::move(__last), std::move(__out), std::forward<_UnaryOperation>(__op));
  }
};

template <class _ExecutionPolicy>
struct __transform_binary<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __out,
             _BinaryOperation&& __op) const noexcept {
    return std::transform(
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        std::move(__out),
        std::forward<_BinaryOperation>(__op));
  }
};

template <class _ExecutionPolicy>
struct __transform_reduce<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&&,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Tp const& __init,
             _BinaryOperation&& __reduce,
             _UnaryOperation&& __transform) const noexcept {
    return std::transform_reduce(
        std::move(__first),
        std::move(__last),
        __init,
        std::forward<_BinaryOperation>(__reduce),
        std::forward<_UnaryOperation>(__transform));
  }
};

template <class _ExecutionPolicy>
struct __transform_reduce_binary<__serial_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation1,
            class _BinaryOperation2>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp> operator()(
      _Policy&&,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Tp const& __init,
      _BinaryOperation1&& __reduce,
      _BinaryOperation2&& __transform) const noexcept {
    return std::transform_reduce(
        std::move(__first1),
        std::move(__last1),
        std::move(__first2),
        std::move(__last2),
        __init,
        std::forward<_BinaryOperation1>(__reduce),
        std::forward<_BinaryOperation2>(__transform));
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_BACKENDS_SERIAL_H
