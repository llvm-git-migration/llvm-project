// RUN: %clang_cc1 -std=c++20 -verify=expected %s

namespace dr2847 { // dr2847: 19

template<typename>
void i();

struct A {
  template<typename>
  void f() requires true;

  template<>
  void f<int>() requires true; // expected-error {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true; // expected-error {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<typename>
struct B {
  void f() requires true;

  template<typename>
  void g() requires true;

  template<typename>
  void h() requires true;

  template<>
  void h<int>() requires true; // expected-error {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true; // expected-error {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<>
void B<int>::f() requires true; // expected-error {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

template<>
template<typename T>
void B<int>::g() requires true;

} // namespace dr2847
