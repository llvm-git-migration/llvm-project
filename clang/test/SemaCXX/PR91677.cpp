// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// expected-no-diagnostics

template <typename> struct t1 {
  template <typename>
  struct t2 {};
};

template <typename T>
t1<T>::template t2<T> f1();

void f2() {
  f1<bool>();
}

namespace N {
  template <typename T> struct A {
    struct B {
      template <typename U> struct X {};
      typedef int arg;
    };
    struct C {
      typedef B::template X<B::arg> x;
    };
  };

  template <> struct A<int>::B {
    template <int N> struct X {};
    static const int arg = 0;
  };
}
