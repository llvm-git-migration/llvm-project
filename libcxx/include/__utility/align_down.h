//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_ALIGN_DOWN_H
#define _LIBCPP___UTILITY_ALIGN_DOWN_H

#include <__config>
#include <cstddef>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI inline _Tp* __align_down(size_t __align, _Tp* __ptr) {
  _LIBCPP_ASSERT_UNCATEGORIZED(
      __align >= alignof(_Tp), "Alignment has to be at least as large as the required alignment");
  return reinterpret_cast<_Tp*>(reinterpret_cast<uintptr_t>(__ptr) & ~(__align - 1));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_ALIGN_DOWN_H
