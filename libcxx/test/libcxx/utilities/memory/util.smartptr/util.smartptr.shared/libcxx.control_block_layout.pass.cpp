//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This test makes sure that the control block implementation used for non-array
// types in std::make_shared and std::allocate_shared is ABI compatible with the
// original implementation.
//
// This test is relevant because the implementation of that control block is
// different starting in C++20, a change that was required to implement P0674.

#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <string>
#include <vector>

#include "test_macros.h"

namespace std {
struct __default_init_tag {};
struct __value_init_tag {};

template <class _Tp, int _Idx, bool _CanBeEmptyBase = is_empty<_Tp>::value && !__libcpp_is_final<_Tp>::value>
struct __compressed_pair_elem {
  using _ParamT         = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(__default_init_tag) {}
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(__value_init_tag) : __value_() {}

  template <class _Up, class = __enable_if_t<!is_same<__compressed_pair_elem, __decay_t<_Up> >::value> >
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(_Up&& __u)
      : __value_(std::forward<_Up>(__u)) {}

#ifndef _LIBCPP_CXX03_LANG
  template <class... _Args, size_t... _Indices>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 explicit __compressed_pair_elem(
      piecewise_construct_t, tuple<_Args...> __args, __tuple_indices<_Indices...>)
      : __value_(std::forward<_Args>(std::get<_Indices>(__args))...) {}
#endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference __get() _NOEXCEPT { return __value_; }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const_reference __get() const _NOEXCEPT { return __value_; }

private:
  _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp {
  using _ParamT         = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;
  using __value_type    = _Tp;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem() = default;
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(__default_init_tag) {}
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(__value_init_tag) : __value_type() {}

  template <class _Up, class = __enable_if_t<!is_same<__compressed_pair_elem, __decay_t<_Up> >::value> >
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair_elem(_Up&& __u)
      : __value_type(std::forward<_Up>(__u)) {}

#ifndef _LIBCPP_CXX03_LANG
  template <class... _Args, size_t... _Indices>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17
  __compressed_pair_elem(piecewise_construct_t, tuple<_Args...> __args, __tuple_indices<_Indices...>)
      : __value_type(std::forward<_Args>(std::get<_Indices>(__args))...) {}
#endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 reference __get() _NOEXCEPT { return *this; }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR const_reference __get() const _NOEXCEPT { return *this; }
};

template <class _T1, class _T2>
class __compressed_pair : private __compressed_pair_elem<_T1, 0>, private __compressed_pair_elem<_T2, 1> {
public:
  // NOTE: This static assert should never fire because __compressed_pair
  // is *almost never* used in a scenario where it's possible for T1 == T2.
  // (The exception is std::function where it is possible that the function
  //  object and the allocator have the same type).
  static_assert(
      (!is_same<_T1, _T2>::value),
      "__compressed_pair cannot be instantiated when T1 and T2 are the same type; "
      "The current implementation is NOT ABI-compatible with the previous implementation for this configuration");

  using _Base1 _LIBCPP_NODEBUG = __compressed_pair_elem<_T1, 0>;
  using _Base2 _LIBCPP_NODEBUG = __compressed_pair_elem<_T2, 1>;

  template <bool _Dummy = true,
            class       = __enable_if_t< __dependent_type<is_default_constructible<_T1>, _Dummy>::value &&
                                   __dependent_type<is_default_constructible<_T2>, _Dummy>::value > >
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair()
      : _Base1(__value_init_tag()), _Base2(__value_init_tag()) {}

  template <class _U1, class _U2>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __compressed_pair(_U1&& __t1, _U2&& __t2)
      : _Base1(std::forward<_U1>(__t1)), _Base2(std::forward<_U2>(__t2)) {}

#ifndef _LIBCPP_CXX03_LANG
  template <class... _Args1, class... _Args2>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 explicit __compressed_pair(
      piecewise_construct_t __pc, tuple<_Args1...> __first_args, tuple<_Args2...> __second_args)
      : _Base1(__pc, std::move(__first_args), typename __make_tuple_indices<sizeof...(_Args1)>::type()),
        _Base2(__pc, std::move(__second_args), typename __make_tuple_indices<sizeof...(_Args2)>::type()) {}
#endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename _Base1::reference first() _NOEXCEPT {
    return static_cast<_Base1&>(*this).__get();
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR typename _Base1::const_reference first() const _NOEXCEPT {
    return static_cast<_Base1 const&>(*this).__get();
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename _Base2::reference second() _NOEXCEPT {
    return static_cast<_Base2&>(*this).__get();
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR typename _Base2::const_reference second() const _NOEXCEPT {
    return static_cast<_Base2 const&>(*this).__get();
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR static _Base1* __get_first_base(__compressed_pair* __pair) _NOEXCEPT {
    return static_cast<_Base1*>(__pair);
  }
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR static _Base2* __get_second_base(__compressed_pair* __pair) _NOEXCEPT {
    return static_cast<_Base2*>(__pair);
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void swap(__compressed_pair& __x)
      _NOEXCEPT_(__is_nothrow_swappable<_T1>::value&& __is_nothrow_swappable<_T2>::value) {
    using std::swap;
    swap(first(), __x.first());
    swap(second(), __x.second());
  }
};
} // namespace std

// This is the pre-C++20 implementation of the control block used by non-array
// std::allocate_shared and std::make_shared. We keep it here so that we can
// make sure our implementation is backwards compatible with it forever.
//
// Of course, the class and its methods were renamed, but the size and layout
// of the class should remain the same as the original implementation.
template <class T, class Alloc>
struct OldEmplaceControlBlock
  : std::__shared_weak_count
{
  explicit OldEmplaceControlBlock(Alloc a) : data_(std::move(a), std::__value_init_tag()) { }
  T* get_elem() noexcept { return std::addressof(data_.second()); }
  Alloc* get_alloc() noexcept { return std::addressof(data_.first()); }

private:
  virtual void __on_zero_shared() noexcept {
    // Not implemented
  }

  virtual void __on_zero_shared_weak() noexcept {
    // Not implemented
  }

  std::__compressed_pair<Alloc, T> data_;
};

template <class T, template <class> class Alloc>
void test() {
  using Old = OldEmplaceControlBlock<T, Alloc<T>>;
  using New = std::__shared_ptr_emplace<T, Alloc<T>>;

  static_assert(sizeof(New) == sizeof(Old), "");
  static_assert(alignof(New) == alignof(Old), "");

  // Also make sure each member is at the same offset
  Alloc<T> a;
  Old old(a);
  New new_(a);

  // 1. Check the stored object
  {
    char const* old_elem = reinterpret_cast<char const*>(old.get_elem());
    char const* new_elem = reinterpret_cast<char const*>(new_.__get_elem());
    std::ptrdiff_t old_offset = old_elem - reinterpret_cast<char const*>(&old);
    std::ptrdiff_t new_offset = new_elem - reinterpret_cast<char const*>(&new_);
    assert(new_offset == old_offset && "offset of stored element changed");
  }

  // 2. Check the allocator
  {
    char const* old_alloc = reinterpret_cast<char const*>(old.get_alloc());
    char const* new_alloc = reinterpret_cast<char const*>(new_.__get_alloc());
    std::ptrdiff_t old_offset = old_alloc - reinterpret_cast<char const*>(&old);
    std::ptrdiff_t new_offset = new_alloc - reinterpret_cast<char const*>(&new_);
    assert(new_offset == old_offset && "offset of allocator changed");
  }

  // Make sure both types have the same triviality (that has ABI impact since
  // it determined how objects are passed). Both should be non-trivial.
  static_assert(std::is_trivial<New>::value == std::is_trivial<Old>::value, "");
}

// Object types to store in the control block
struct TrivialEmptyType { };
struct TrivialNonEmptyType { char c[11]; };
struct FinalEmptyType final { };
struct NonTrivialType {
  char c[22];
  NonTrivialType() : c{'x'} { }
};

// Allocator types
template <class T>
struct TrivialEmptyAlloc {
  using value_type = T;
  TrivialEmptyAlloc() = default;
  template <class U> TrivialEmptyAlloc(TrivialEmptyAlloc<U>) { }
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) { }
};
template <class T>
struct TrivialNonEmptyAlloc {
  char storage[77];
  using value_type = T;
  TrivialNonEmptyAlloc() = default;
  template <class U> TrivialNonEmptyAlloc(TrivialNonEmptyAlloc<U>) { }
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) { }
};
template <class T>
struct FinalEmptyAlloc final {
  using value_type = T;
  FinalEmptyAlloc() = default;
  template <class U> FinalEmptyAlloc(FinalEmptyAlloc<U>) { }
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) { }
};
template <class T>
struct NonTrivialAlloc {
  char storage[88];
  using value_type = T;
  NonTrivialAlloc() { }
  template <class U> NonTrivialAlloc(NonTrivialAlloc<U>) { }
  T* allocate(std::size_t) { return nullptr; }
  void deallocate(T*, std::size_t) { }
};

int main(int, char**) {
  test<TrivialEmptyType, TrivialEmptyAlloc>();
  test<TrivialEmptyType, TrivialNonEmptyAlloc>();
  test<TrivialEmptyType, FinalEmptyAlloc>();
  test<TrivialEmptyType, NonTrivialAlloc>();

  test<TrivialNonEmptyType, TrivialEmptyAlloc>();
  test<TrivialNonEmptyType, TrivialNonEmptyAlloc>();
  test<TrivialNonEmptyType, FinalEmptyAlloc>();
  test<TrivialNonEmptyType, NonTrivialAlloc>();

  test<FinalEmptyType, TrivialEmptyAlloc>();
  test<FinalEmptyType, TrivialNonEmptyAlloc>();
  test<FinalEmptyType, FinalEmptyAlloc>();
  test<FinalEmptyType, NonTrivialAlloc>();

  test<NonTrivialType, TrivialEmptyAlloc>();
  test<NonTrivialType, TrivialNonEmptyAlloc>();
  test<NonTrivialType, FinalEmptyAlloc>();
  test<NonTrivialType, NonTrivialAlloc>();

  // Test a few real world types just to make sure we didn't mess up badly somehow
  test<std::string, std::allocator>();
  test<int, std::allocator>();
  test<std::vector<int>, std::allocator>();

  return 0;
}
