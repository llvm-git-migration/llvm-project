// RUN: %clang_cc1 -verify %s -std=c++11 -pedantic-errors

enum class E;

template<typename T>
struct A {
  enum class F;
};

struct B {
  template<typename T>
  friend enum A<T>::F; // expected-error {{elaborated enumeration type cannot be a friend}}

  // FIXME: Per [temp.expl.spec]p19, a friend declaration cannot be an explicit specialization
  template<>
  friend enum A<int>::F; // expected-error {{elaborated enumeration type cannot be a friend}}

  enum class G;

  friend enum E; // expected-error {{elaborated enumeration type cannot be a friend}}
};

template<typename T>
struct C {
  friend enum T::G; // expected-error {{elaborated enumeration type cannot be a friend}}
  friend enum A<T>::G; // expected-error {{elaborated enumeration type cannot be a friend}}
};

struct D {
  friend enum B::G; // expected-error {{elaborated enumeration type cannot be a friend}}
  friend enum class B::G; // expected-error {{elaborated enumeration type cannot be a friend}}
                          // expected-error@-1 {{reference to enumeration must use 'enum' not 'enum class'}}
};
