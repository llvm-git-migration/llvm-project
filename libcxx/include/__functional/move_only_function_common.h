//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_COMMON_H
#define _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_COMMON_H

#include <__config>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class...>
class move_only_function;

template <class>
struct __is_move_only_function : false_type {};

template <class... _Args>
struct __is_move_only_function<move_only_function<_Args...>> : true_type {};

template <class _BufferT, class _ReturnT, class... _ArgTypes>
struct _MoveOnlyFunctionTrivialVTable {
  using _CallFunc = _ReturnT(_BufferT&, _ArgTypes...);

  _CallFunc* __call_;
};

template <class _BufferT, class _ReturnT, class... _ArgTypes>
struct _MoveOnlyFunctionNonTrivialVTable : _MoveOnlyFunctionTrivialVTable<_BufferT, _ReturnT, _ArgTypes...> {
  using _DestroyFunc = void(_BufferT&) noexcept;

  _DestroyFunc* __destroy_;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_COMMON_H
