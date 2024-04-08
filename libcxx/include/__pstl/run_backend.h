//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_RUN_BACKEND_H
#define _LIBCPP___PSTL_RUN_BACKEND_H

#include <__config>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <new> // __throw_bad_alloc
#include <optional>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _BackendFunction, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE auto __run_backend_impl(_Args&&... __args) noexcept {
  return _BackendFunction{}(std::forward<_Args>(__args)...);
}

// This function is used to call a backend PSTL algorithm from a frontend algorithm.
//
// All PSTL backend algorithms return an optional denoting whether there was an
// "infrastructure"-level failure (aka failure to allocate). This function takes
// care of unwrapping that and throwing `bad_alloc()` in case there was a problem
// in the underlying implementation.
//
// We must also be careful not to call any user code that could throw an exception
// (such as moving or copying iterators) in here since that should terminate the
// program, which is why we delegate to a noexcept helper below.
template <class _BackendFunction, class... _Args>
_LIBCPP_HIDE_FROM_ABI auto __run_backend(_Args&&... __args) {
  auto __result = __pstl::__run_backend_impl<_BackendFunction>(std::forward<_Args>(__args)...);
  if (__result == nullopt)
    std::__throw_bad_alloc();
  else
    return std::move(*__result);
}

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___PSTL_RUN_BACKEND_H
