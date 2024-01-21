// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_BOUNDED_ITER_H
#define _LIBCPP___ITERATOR_BOUNDED_ITER_H

#include <__assert>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_convertible.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// Iterator wrapper that carries the valid range it is allowed to access.
//
// This is a simple iterator wrapper for contiguous iterators that points
// within a [begin, end] range and carries these bounds with it. The iterator
// ensures that it is pointing within [begin, end) range when it is
// dereferenced. It also ensures that it is never iterated outside of
// [begin, end].
template <class _Iterator, class = __enable_if_t< __libcpp_is_contiguous_iterator<_Iterator>::value > >
struct __bounded_iter {
  using value_type        = typename iterator_traits<_Iterator>::value_type;
  using difference_type   = typename iterator_traits<_Iterator>::difference_type;
  using pointer           = typename iterator_traits<_Iterator>::pointer;
  using reference         = typename iterator_traits<_Iterator>::reference;
  using iterator_category = typename iterator_traits<_Iterator>::iterator_category;
#if _LIBCPP_STD_VER >= 20
  using iterator_concept = contiguous_iterator_tag;
#endif

  // Create a singular iterator.
  //
  // Such an iterator does not point to any object and is conceptually out of bounds, so it is
  // not dereferenceable. Observing operations like comparison and assignment are valid.
  _LIBCPP_HIDE_FROM_ABI __bounded_iter() = default;

  _LIBCPP_HIDE_FROM_ABI __bounded_iter(__bounded_iter const&) = default;
  _LIBCPP_HIDE_FROM_ABI __bounded_iter(__bounded_iter&&)      = default;

  template <class _OtherIterator, class = __enable_if_t< is_convertible<_OtherIterator, _Iterator>::value > >
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __bounded_iter(__bounded_iter<_OtherIterator> const& __other) _NOEXCEPT
      : __current_(__other.__current_),
        __begin_(__other.__begin_),
        __end_(__other.__end_) {}

  // Assign a bounded iterator to another one, rebinding the bounds of the iterator as well.
  _LIBCPP_HIDE_FROM_ABI __bounded_iter& operator=(__bounded_iter const&) = default;
  _LIBCPP_HIDE_FROM_ABI __bounded_iter& operator=(__bounded_iter&&)      = default;

private:
  // Create an iterator wrapping the given iterator, and whose bounds are described
  // by the provided [begin, end] range.
  //
  // Except in debug builds, the constructor does not check whether the resulting iterator
  // is within its bounds. The container is assumed to be self-consistent.
  //
  // Since it is non-standard for iterators to have this constructor, __bounded_iter must
  // be created via `std::__make_bounded_iter`.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 explicit __bounded_iter(
      _Iterator __current, _Iterator __begin, _Iterator __end)
      : __current_(__current), __begin_(__begin), __end_(__end) {
    _LIBCPP_ASSERT_INTERNAL(__begin <= __current, "__bounded_iter(current, begin, end): current and begin are inconsistent");
    _LIBCPP_ASSERT_INTERNAL(__current <= __end, "__bounded_iter(current, begin, end): current and end are inconsistent");
  }

  template <class _It>
  friend _LIBCPP_CONSTEXPR __bounded_iter<_It> __make_bounded_iter(_It, _It, _It);

public:
  // Dereference and indexing operations.
  //
  // These operations check that the iterator is dereferenceable, that is within [begin, end).
  // The iterator is always within [begin, end], so we only need to check for end. This is
  // easier to optimize out. See https://github.com/llvm/llvm-project/issues/78829
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference operator*() const _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __current_ != __end_, "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
    return *__current_;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pointer operator->() const _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __current_ != __end_, "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
    return std::__to_address(__current_);
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference operator[](difference_type __n) const _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n >= __begin_ - __current_, "__bounded_iter::operator[]: Attempt to index an iterator past the start");
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n < __end_ - __current_, "__bounded_iter::operator[]: Attempt to index an iterator past the end");
    return __current_[__n];
  }

  // Arithmetic operations.
  //
  // These operations do not check that the resulting iterator is within the bounds, since that
  // would make it impossible to create a past-the-end iterator.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter& operator++() _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __current_ != __end_, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    ++__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter operator++(int) _NOEXCEPT {
    __bounded_iter __tmp(*this);
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter& operator--() _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __current_ != __begin_, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    --__current_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter operator--(int) _NOEXCEPT {
    __bounded_iter __tmp(*this);
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter& operator+=(difference_type __n) _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n >= __begin_ - __current_, "__bounded_iter::operator+=: Attempt to rewind an iterator past the start");
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n <= __end_ - __current_, "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    __current_ += __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __bounded_iter
  operator+(__bounded_iter const& __self, difference_type __n) _NOEXCEPT {
    __bounded_iter __tmp(__self);
    __tmp += __n;
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __bounded_iter
  operator+(difference_type __n, __bounded_iter const& __self) _NOEXCEPT {
    __bounded_iter __tmp(__self);
    __tmp += __n;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 __bounded_iter& operator-=(difference_type __n) _NOEXCEPT {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n <= __current_ - __begin_, "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __n >= __current_ - __end_, "__bounded_iter::operator-=: Attempt to advance an iterator past the end");
    __current_ -= __n;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 friend __bounded_iter
  operator-(__bounded_iter const& __self, difference_type __n) _NOEXCEPT {
    __bounded_iter __tmp(__self);
    __tmp -= __n;
    return __tmp;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 friend difference_type
  operator-(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ - __y.__current_;
  }

  // Comparison operations.
  //
  // These operations do not check whether the iterators are within their bounds.
  // The valid range for each iterator is also not considered as part of the comparison,
  // i.e. two iterators pointing to the same location will be considered equal even
  // if they have different validity ranges.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator==(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ == __y.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator!=(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ != __y.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator<(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ < __y.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator>(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ > __y.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator<=(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ <= __y.__current_;
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR friend bool
  operator>=(__bounded_iter const& __x, __bounded_iter const& __y) _NOEXCEPT {
    return __x.__current_ >= __y.__current_;
  }

private:
  template <class>
  friend struct pointer_traits;
  _Iterator __current_;       // current iterator
  _Iterator __begin_, __end_; // valid range represented as [begin, end]
};

template <class _It>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __bounded_iter<_It> __make_bounded_iter(_It __it, _It __begin, _It __end) {
  return __bounded_iter<_It>(std::move(__it), std::move(__begin), std::move(__end));
}

#if _LIBCPP_STD_VER <= 17
template <class _Iterator>
struct __libcpp_is_contiguous_iterator<__bounded_iter<_Iterator> > : true_type {};
#endif

template <class _Iterator>
struct pointer_traits<__bounded_iter<_Iterator> > {
  using pointer         = __bounded_iter<_Iterator>;
  using element_type    = typename pointer_traits<_Iterator>::element_type;
  using difference_type = typename pointer_traits<_Iterator>::difference_type;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR static element_type* to_address(pointer __it) _NOEXCEPT {
    return std::__to_address(__it.__current_);
  }
};

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ITERATOR_BOUNDED_ITER_H
