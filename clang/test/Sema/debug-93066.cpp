// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

struct S {
  void f() {
    ++this; // expected-error {{expression is not assignable}}
    // expected-note@-1 {{add '*' to dereference it}}
  }

  void g() const {
    ++this; // expected-error {{expression is not assignable}}
  }
};

void f(int* a, int* const b, const int* const c, __UINTPTR_TYPE__ d) {
  (int*)d = 4; // expected-error {{assignment to cast is illegal, lvalue casts are not supported}}
  // expected-note@-1 {{add '*' to dereference it}}
    
  ++a;
  ++b; // expected-error {{cannot assign to variable 'b' with const-qualified type 'int *const'}}
  // expected-note@-1 {{add '*' to dereference it}}
  // expected-note@* {{variable 'b' declared const here}}
  ++c; // expected-error {{cannot assign to variable 'c' with const-qualified type 'const int *const'}}
  // expected-note@* {{variable 'c' declared const here}}

  reinterpret_cast<int*>(42) += 3; // expected-error {{expression is not assignable}}
  // expected-note@-1 {{add '*' to dereference it}}
    
  const int x = 42;
  (const_cast<int*>(&x)) += 3; // expected-error {{expression is not assignable}}
  // expected-note@-1 {{add '*' to dereference it}}
}

template <typename T>
void f(T& t) {
    // expected-note@* 2 {{variable 't' declared const here}}
    ++t;
    // expected-error@* {{cannot assign to variable 't' with const-qualified type 'int *const &'}}
    // expected-error@* {{cannot assign to variable 't' with const-qualified type 'const int *const &'}}
    // expected-note@* {{add '*' to dereference it}}
}

void g() {
    int* a;
    int* const b = a;
    const int* const c = a;
    f(a);
    f(b); // expected-note {{in instantiation of function template specialization 'f<int *const>' requested here}}
    f(c); // expected-note {{in instantiation of function template specialization 'f<const int *const>' requested here}}
}
