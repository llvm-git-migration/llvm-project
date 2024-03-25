//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_ALIASING_ITERATOR_H
#define _LIBCPP___ITERATOR_ALIASING_ITERATOR_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_trivial.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _BaseT, class _Alias>
struct __aliasing_iterator_wrapper {
  class __iterator {
    _BaseT* __base_ = nullptr;

  public:
    using iterator_category = random_access_iterator_tag;
    using value_type        = _Alias;
    using difference_type   = ptrdiff_t;

    static_assert(is_trivial<_Alias>::value);
    static_assert(sizeof(_BaseT) == sizeof(_Alias));

    _LIBCPP_HIDE_FROM_ABI __iterator() = default;
    _LIBCPP_HIDE_FROM_ABI __iterator(_BaseT* __base) _NOEXCEPT : __base_(__base) {}

    _LIBCPP_HIDE_FROM_ABI __iterator& operator++() _NOEXCEPT {
      ++__base_;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator++(int) _NOEXCEPT {
      __iterator __tmp(*this);
      ++__base_;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator& operator--() _NOEXCEPT {
      --__base_;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator--(int) _NOEXCEPT {
      __iterator __tmp(*this);
      --__base_;
      return __tmp;
    }

    _LIBCPP_HIDE_FROM_ABI friend __iterator operator+(__iterator __iter, difference_type __n) _NOEXCEPT {
      return __iterator(__iter.__base_ + __n);
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator+=(difference_type __n) _NOEXCEPT {
      __base_ += __n;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI __iterator operator-(difference_type __n) _NOEXCEPT { return __iterator(__base_ - __n); }

    _LIBCPP_HIDE_FROM_ABI __iterator operator-=(difference_type __n) _NOEXCEPT {
      __base_ -= __n;
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI friend difference_type operator-(__iterator __lhs, __iterator __rhs) _NOEXCEPT {
      return __lhs.__base_ - __rhs.__base_;
    }

    _LIBCPP_HIDE_FROM_ABI _BaseT* base() _NOEXCEPT { return __base_; }
    _LIBCPP_HIDE_FROM_ABI const _BaseT* base() const _NOEXCEPT { return __base_; }

    _LIBCPP_HIDE_FROM_ABI _Alias operator*() const _NOEXCEPT {
      _Alias __val;
      __builtin_memcpy(&__val, __base_, sizeof(_BaseT));
      return __val;
    }

    _LIBCPP_HIDE_FROM_ABI _Alias operator[](difference_type __n) const _NOEXCEPT { return *(*this + __n); }

    _LIBCPP_HIDE_FROM_ABI friend bool operator==(const __iterator& __lhs, const __iterator& __rhs) _NOEXCEPT {
      return __lhs.__base_ == __rhs.__base_;
    }

    _LIBCPP_HIDE_FROM_ABI friend bool operator!=(const __iterator& __lhs, const __iterator& __rhs) _NOEXCEPT {
      return __lhs.__base_ != __rhs.__base_;
    }
  };
};

// This is required to avoid ADL instantiations on _BaseT
template <class _BaseT, class _Alias>
using __aliasing_iterator = __aliasing_iterator_wrapper<_BaseT, _Alias>::__iterator;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_ALIASING_ITERATOR_H
