//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// constexpr variant(variant&&) noexcept(see below);

#include <cassert>
#include <string>
#include <type_traits>
#include <variant>

#include "test_macros.h"
#include "test_workarounds.h"

struct ThrowsMove {
  ThrowsMove(ThrowsMove&&) noexcept(false) {}
};

struct NoCopy {
  NoCopy(const NoCopy&) = delete;
};

struct MoveOnly {
  int value;
  constexpr MoveOnly(int v) : value(v) {}
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&)      = default;
};

struct MoveOnlyNT {
  int value;
  constexpr MoveOnlyNT(int v) : value(v) {}
  MoveOnlyNT(const MoveOnlyNT&) = delete;
  constexpr MoveOnlyNT(MoveOnlyNT&& other) : value(other.value) { other.value = -1; }
};

struct NTMove {
  constexpr NTMove(int v) : value(v) {}
  NTMove(const NTMove&) = delete;
  NTMove(NTMove&& that) : value(that.value) { that.value = -1; }
  int value;
};

static_assert(!std::is_trivially_move_constructible<NTMove>::value, "");
static_assert(std::is_move_constructible<NTMove>::value, "");

struct TMove {
  constexpr TMove(int v) : value(v) {}
  TMove(const TMove&) = delete;
  TMove(TMove&&)      = default;
  int value;
};

static_assert(std::is_trivially_move_constructible<TMove>::value, "");

struct TMoveNTCopy {
  constexpr TMoveNTCopy(int v) : value(v) {}
  TMoveNTCopy(const TMoveNTCopy& that) : value(that.value) {}
  TMoveNTCopy(TMoveNTCopy&&) = default;
  int value;
};

static_assert(std::is_trivially_move_constructible<TMoveNTCopy>::value, "");

#ifndef TEST_HAS_NO_EXCEPTIONS
struct MakeEmptyT {
  static int alive;
  MakeEmptyT() { ++alive; }
  MakeEmptyT(const MakeEmptyT&) {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT&&) { throw 42; }
  MakeEmptyT& operator=(const MakeEmptyT&) { throw 42; }
  MakeEmptyT& operator=(MakeEmptyT&&) { throw 42; }
  ~MakeEmptyT() { --alive; }
};

int MakeEmptyT::alive = 0;

template <class Variant>
void makeEmpty(Variant& v) {
  Variant v2(std::in_place_type<MakeEmptyT>);
  try {
    v = std::move(v2);
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

void test_move_noexcept() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnly>;
    static_assert(std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnlyNT>;
    static_assert(!std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, ThrowsMove>;
    static_assert(!std::is_nothrow_move_constructible<V>::value, "");
  }
}

void test_move_ctor_sfinae() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnly>;
    static_assert(std::is_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, MoveOnlyNT>;
    static_assert(std::is_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NoCopy>;
    static_assert(!std::is_move_constructible<V>::value, "");
  }

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = std::variant<int, long>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NTMove>;
    static_assert(!std::is_trivially_move_constructible<V>::value, "");
    static_assert(std::is_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TMove>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TMoveNTCopy>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
}

template <typename T>
struct Result {
  std::size_t index;
  T value;
};

TEST_CONSTEXPR_CXX20 bool test_move_ctor_basic() {
  {
    std::variant<int> v(std::in_place_index<0>, 42);
    std::variant<int> v2 = std::move(v);
    assert(v2.index() == 0);
    assert(std::get<0>(v2) == 42);
  }
  {
    std::variant<int, long> v(std::in_place_index<1>, 42);
    std::variant<int, long> v2 = std::move(v);
    assert(v2.index() == 1);
    assert(std::get<1>(v2) == 42);
  }
  {
    std::variant<MoveOnly> v(std::in_place_index<0>, 42);
    assert(v.index() == 0);
    std::variant<MoveOnly> v2(std::move(v));
    assert(v2.index() == 0);
    assert(std::get<0>(v2).value == 42);
  }
  {
    std::variant<int, MoveOnly> v(std::in_place_index<1>, 42);
    assert(v.index() == 1);
    std::variant<int, MoveOnly> v2(std::move(v));
    assert(v2.index() == 1);
    assert(std::get<1>(v2).value == 42);
  }
  {
    std::variant<MoveOnlyNT> v(std::in_place_index<0>, 42);
    assert(v.index() == 0);
    std::variant<MoveOnlyNT> v2(std::move(v));
    assert(v2.index() == 0);
    assert(std::get<0>(v).value == -1);
    assert(std::get<0>(v2).value == 42);
  }
  {
    std::variant<int, MoveOnlyNT> v(std::in_place_index<1>, 42);
    assert(v.index() == 1);
    std::variant<int, MoveOnlyNT> v2(std::move(v));
    assert(v2.index() == 1);
    assert(std::get<1>(v).value == -1);
    assert(std::get<1>(v2).value == 42);
  }

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct {
      constexpr Result<int> operator()() const {
        std::variant<int> v(std::in_place_index<0>, 42);
        std::variant<int> v2 = std::move(v);
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value == 42, "");
  }
  {
    struct {
      constexpr Result<long> operator()() const {
        std::variant<int, long> v(std::in_place_index<1>, 42);
        std::variant<int, long> v2 = std::move(v);
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMove> operator()() const {
        std::variant<TMove> v(std::in_place_index<0>, 42);
        std::variant<TMove> v2(std::move(v));
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMove> operator()() const {
        std::variant<int, TMove> v(std::in_place_index<1>, 42);
        std::variant<int, TMove> v2(std::move(v));
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMoveNTCopy> operator()() const {
        std::variant<TMoveNTCopy> v(std::in_place_index<0>, 42);
        std::variant<TMoveNTCopy> v2(std::move(v));
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMoveNTCopy> operator()() const {
        std::variant<int, TMoveNTCopy> v(std::in_place_index<1>, 42);
        std::variant<int, TMoveNTCopy> v2(std::move(v));
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
  return true;
}

void test_move_ctor_valueless_by_exception() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = std::variant<int, MakeEmptyT>;
  V v1;
  makeEmpty(v1);
  V v(std::move(v1));
  assert(v.valueless_by_exception());
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <std::size_t Idx, class T>
constexpr bool test_constexpr_ctor_imp(const T& v) {
  auto copy = v;
  auto v2   = std::move(copy);
  return v2.index() == v.index() && v2.index() == Idx && std::get<Idx>(v2) == std::get<Idx>(v);
}

#if TEST_STD_VER >= 20
struct NonTrivialMoveCtor {
  int i = 0;
  constexpr NonTrivialMoveCtor(int ii) : i(ii) {}
  constexpr NonTrivialMoveCtor(const NonTrivialMoveCtor& other) = default;
  constexpr NonTrivialMoveCtor(NonTrivialMoveCtor&& other) : i(other.i) {}
  constexpr ~NonTrivialMoveCtor()                                                        = default;
  friend constexpr bool operator==(const NonTrivialMoveCtor&, const NonTrivialMoveCtor&) = default;
};
#endif

void test_constexpr_move_ctor() {
  {
    // Test is_trivially_move_constructible
    // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
    using V = std::variant<long, void*, const int>;
#ifdef TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
    static_assert(std::is_trivially_destructible<V>::value, "");
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
    static_assert(std::is_trivially_move_constructible<V>::value, "");
    static_assert(!std::is_copy_assignable<V>::value, "");
    static_assert(!std::is_move_assignable<V>::value, "");
#else  // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
    static_assert(std::is_trivially_copyable<V>::value, "");
#endif // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
    static_assert(std::is_trivially_move_constructible<V>::value, "");
    static_assert(test_constexpr_ctor_imp<0>(V(42l)), "");
    static_assert(test_constexpr_ctor_imp<1>(V(nullptr)), "");
    static_assert(test_constexpr_ctor_imp<2>(V(101)), "");
  }

#if TEST_STD_VER >= 20
  {
    // Test !is_trivially_move_constructible
    using V = std::variant<long, NonTrivialMoveCtor, void*>;
    static_assert(!std::is_trivially_move_constructible<V>::value, "");
    static_assert(test_constexpr_ctor_imp<0>(V(42l)), "");
    static_assert(test_constexpr_ctor_imp<1>(V(NonTrivialMoveCtor(5))), "");
    static_assert(test_constexpr_ctor_imp<2>(V(nullptr)), "");
  }
#endif
}

int main(int, char**) {
  test_move_ctor_basic();
  test_move_ctor_valueless_by_exception();
  test_move_noexcept();
  test_move_ctor_sfinae();
  test_constexpr_move_ctor();

#if TEST_STD_VER >= 20
  static_assert(test_move_ctor_basic());
#endif

  return 0;
}
