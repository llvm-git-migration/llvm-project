// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_COMPRESSED_PAIR_H
#define _LIBCPP___MEMORY_COMPRESSED_PAIR_H

#include <__config>
#include <__type_traits/datasizeof.h>
#include <__type_traits/is_empty.h>
#include <__type_traits/is_final.h>
#include <__type_traits/is_object.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_ABI_NO_COMPRESSED_PAIR_PADDING

template <class _ToPad>
class __compressed_pair_padding {
  char __padding_[(!is_object<_ToPad>::value || (is_empty<_ToPad>::value && !__libcpp_is_final<_ToPad>::value))
                      ? 0
                      : sizeof(_ToPad) - __libcpp_datasizeof<_ToPad>::value];
};

#  define _LIBCPP_COMPRESSED_PAIR_PADDING(T, Name) _LIBCPP_NO_UNIQUE_ADDRESS __compressed_pair_padding<T> Name
#  define _LIBCPP_COMPRESSED_PAIR(T1, Name1, T2, Name2)                                                                \
    [[__gnu__::__aligned__(_LIBCPP_ALIGNOF(T2))]] _LIBCPP_NO_UNIQUE_ADDRESS T1 Name1;                                  \
    _LIBCPP_COMPRESSED_PAIR_PADDING(T1, Name1##_padding_);                                                             \
    _LIBCPP_NO_UNIQUE_ADDRESS T2 Name2;                                                                                \
    _LIBCPP_COMPRESSED_PAIR_PADDING(T2, Name2##_padding_)
#else
#  define _LIBCPP_COMPRESSED_PAIR_PADDING(T, Name)
#  define _LIBCPP_COMPRESSED_PAIR(T1, Name1, T2, Name2)                                                                \
    _LIBCPP_NO_UNIQUE_ADDRESS T1 Name1;                                                                                \
    _LIBCPP_NO_UNIQUE_ADDRESS T2 Name2;
#endif // _LIBCPP_ABI_NO_COMPRESSED_PAIR_PADDING

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_COMPRESSED_PAIR_H
