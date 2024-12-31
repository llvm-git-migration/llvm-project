// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++23 %s -verify

struct A {};
using CA = const A;

struct S1 : CA {           // expected-warning {{'const' qualifier on base class type 'CA' (aka 'const A') have no effect}} \
                           // expected-note {{base class 'CA' (aka 'const A') specified here}}
  constexpr S1() : CA() {}
};

struct S2 : A {
  constexpr S2() : CA() {}
};

struct S3 : CA {          // expected-warning {{'const' qualifier on base class type 'CA' (aka 'const A') have no effect}} \
                          // expected-note {{base class 'CA' (aka 'const A') specified here}}
  constexpr S3() : A() {}
};

struct Int {};

template <class _Hp>
struct __tuple_leaf : _Hp {           // expected-warning {{'const' qualifier on base class type 'const Int' have no effect}} \
                                      // expected-note {{base class 'const Int' specified here}}
  constexpr __tuple_leaf() : _Hp() {}
};

constexpr __tuple_leaf<const Int> t;  // expected-note {{in instantiation of template class '__tuple_leaf<const Int>' requested here}}
