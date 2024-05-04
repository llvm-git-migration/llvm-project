//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
#define _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H

#include <__algorithm/ranges_find_if.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/projected.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__ranges/dangling.h>
#include <__ranges/subrange.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {

template <typename _I, typename _S, typename _T, typename _P>
_LIBCPP_HIDE_FROM_ABI constexpr subrange<_I> __find_last_impl(_I __first, _S __last, const _T& __value, _P& __proj) {
  _I __ret{};
  for (; __first != __last; ++__first)
    if (std::invoke(__proj, *__first) == __value)
      __ret = __first;

  if (__ret == _I{})
    return {__first, __first};

  return {__ret, std::ranges::next(__ret, __last)};
}

namespace __find_last {
struct __fn {
  template <forward_iterator _I, sentinel_for<_I> _S, typename _T, typename _Proj = identity>
    requires indirect_binary_predicate<equal_to, projected<_I, _Proj>, const _T*>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr subrange<_I>
  operator()(_I __first, _S __last, const _T& __value, _Proj __proj = {}) const {
    return __find_last_impl(std::move(__first), std::move(__last), __value, __proj);
  }

  template <forward_range _R, typename _T, typename _P = identity>
    requires indirect_binary_predicate<equal_to, projected<iterator_t<_R>, _P>, const _T*>
  constexpr ranges::borrowed_subrange_t<_R> operator()(_R&& __r, const _T& __value, _P __proj = {}) const {
    return this->operator()(begin(__r), end(__r), __value, ref(__proj));
  }
};

} // namespace __find_last

inline namespace __cpo {
inline constexpr __find_last::__fn find_last{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_RANGES_FIND_LAST_H
