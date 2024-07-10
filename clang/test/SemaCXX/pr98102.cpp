// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

template <bool v>
struct BC {
  static constexpr bool value = v;
};

template <typename B>
struct A : B {
  static constexpr bool value = B::value;
};

template <typename T>
using _Requires = A<T>::value;

template <typename _Tp, typename _Args>
struct __is_constructible_impl : BC<__is_constructible(_Tp, _Args)> {};

template <typename _Tp>
struct optional {
  template <typename _Up, _Requires<__is_constructible_impl<_Tp, _Up>> = true>
  optional(_Up) {}
};

struct MO {};
struct S : MO {};
struct TB {
  TB(optional<S>) {}
};

class TD : public TB, MO {
  using TB::TB;
};

void foo() {
  static_assert(__is_constructible_impl<TD, TD>::value);
}
