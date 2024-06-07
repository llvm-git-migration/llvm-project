// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename ...T> struct X {
  void f(int);
  void f(...); // expected-note {{member is declared here}} \
                  expected-note {{member is declared here}}
  static int n;
};

template<typename T, typename U> using A = T;

// These definitions are OK, X<A<T, decltype(...)>...> is equivalent to X<T...>
// so this defines the member of the primary template.
template<typename ...T>
void X<A<T, decltype(f(T()))>...>::f(int) {} // expected-error {{explicit qualification required to use member 'f' from dependent base class}} \
                                                expected-error {{call to non-static member function without an object argument}}
template<typename ...T>
int X<A<T, decltype(f(T()))>...>::n = 0; // expected-error {{explicit qualification required to use member 'f' from dependent base class}} \
                                            expected-error {{call to non-static member function without an object argument}}
struct Y {}; void f(Y);

void g() {
  // OK, substitution succeeds.
  X<Y>().f(0);
  X<Y>::n = 1;

  // Error, substitution fails; this should not be treated as a SFINAE-able
  // condition, so we don't select X<void>::f(...).
  X<void>().f(0); // expected-note {{in instantiation of member function 'X<void>::f' requested here}}
  X<void>::n = 1; // expected-note {{in instantiation of static data member 'X<void>::n' requested here}}
}
