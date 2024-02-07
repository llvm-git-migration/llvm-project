//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_FUNCTION_OBJECTS_FUNC_BIND_PARTIAL_TYPES_H
#define TEST_STD_UTILITIES_FUNCTION_OBJECTS_FUNC_BIND_PARTIAL_TYPES_H

#include <tuple>
#include <utility>

struct CopyMoveInfo {
  enum { none, copy, move } copy_kind;

  constexpr CopyMoveInfo() : copy_kind(none) {}
  constexpr CopyMoveInfo(CopyMoveInfo const&) : copy_kind(copy) {}
  constexpr CopyMoveInfo(CopyMoveInfo&&) : copy_kind(move) {}
};

struct NotCopyMove {
  NotCopyMove()                   = delete;
  NotCopyMove(const NotCopyMove&) = delete;
  NotCopyMove(NotCopyMove&&)      = delete;
  template <class... Args>
  void operator()(Args&&...) const {}
};

struct NonConstCopyConstructible {
  explicit NonConstCopyConstructible() {}
  NonConstCopyConstructible(NonConstCopyConstructible&) {}
};

struct MoveConstructible {
  explicit MoveConstructible() {}
  MoveConstructible(MoveConstructible&&) {}
};

struct MakeTuple {
  template <class... Args>
  constexpr auto operator()(Args&&... args) const {
    return std::make_tuple(std::forward<Args>(args)...);
  }
};

template <int X>
struct Elem {
  template <int Y>
  constexpr bool operator==(Elem<Y> const&) const {
    return X == Y;
  }
};

struct NotMoveConst {
  NotMoveConst(NotMoveConst&&)      = delete;
  NotMoveConst(NotMoveConst const&) = delete;

  NotMoveConst(int) {}
};

constexpr int pass(const int n) { return n; }

inline int simple(int n) { return n; }

template <class T>
T do_nothing(T t) {
  return t;
}

inline void testNotMoveConst(NotMoveConst) {}

#endif // TEST_STD_UTILITIES_FUNCTION_OBJECTS_FUNC_BIND_PARTIAL_TYPES_H
