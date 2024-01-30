//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALLOCATION_RESULT_H
#define _LIBCPP___MEMORY_ALLOCATION_RESULT_H

#include <__config>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23
#  ifndef _LIBCPP_ENABLE_CXX23_USER_SPECIALIZATION_OF_ALLOCATOR_TRAITS

template <class _Pointer, class SizeType = size_t>
struct allocation_result {
  _Pointer ptr;
  SizeType count;
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(allocation_result);

#  else

template <class _Pointer>
struct allocation_result {
  _Pointer ptr;
  size_t count;
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(allocation_result);

#  endif // _LIBCPP_ENABLE_CXX23_USER_SPECIALIZATION_OF_ALLOCATOR_TRAITS
#endif   // _LIBCPP_STD_VER

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATION_RESULT_H
