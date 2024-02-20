// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

struct A {
  static constexpr bool x = true;
};

template<typename T, typename U>
void f(T, U) noexcept(T::x);

template<typename T, typename U>
void f(T, U*) noexcept(T::x);

template<typename T, typename U>
void f(T, U**) noexcept(T::y); // expected-error {{no member named 'y' in 'A'}}

template<typename T, typename U>
void f(T, U***) noexcept(T::x);

template<>
void f(A, int*) noexcept; // expected-note {{previous declaration is here}}

template<>
void f(A, int*); // expected-error {{'f<A, int>' is missing exception specification 'noexcept'}}

template<>
void f(A, int**) noexcept; // expected-error {{exception specification in declaration does not match previous declaration}}
                           // expected-note@-1 {{in instantiation of exception specification for 'f<A, int>' requested here}}
                           // expected-note@-2 {{previous declaration is here}}

// FIXME: Exception specification is currently set to EST_None if instantiation fails.
template<>
void f(A, int**);

template<>
void f(A, int***) noexcept; // expected-note {{previous declaration is here}}

template<>
void f(A, int***); // expected-error {{'f<A, int>' is missing exception specification 'noexcept'}}
