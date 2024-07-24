// RUN: %clang_cc1 -std=c++23 -verify %s

template<bool> struct enable_if { typedef void type; };
template <class T> class Foo {};
template <class X> constexpr bool check() { return true; }
template <class X, class Enable = void> struct Bar {};

template<class X> void func(Bar<X, typename enable_if<check<X>()>::type>) {}
// expected-note@-1 {{candidate function}}

template<class T> void func(Bar<Foo<T>>) {}
// expected-note@-1 {{candidate function}}

void g() {
  func(Bar<Foo<int>>()); // expected-error {{call to 'func' is ambiguous}}
}
