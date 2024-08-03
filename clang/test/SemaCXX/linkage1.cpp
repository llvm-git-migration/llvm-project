// RUN: %clang_cc1 -fsyntax-only -verify -pedantic-errors %s

namespace c {
  extern "C" void main(); // expected-error {{'main' cannot have linkage specification 'extern "C"'}}
}
extern "C" {
  int main(); // expected-error {{'main' cannot have linkage specification 'extern "C"'}}
}

extern "C" int main(); // expected-error {{'main' cannot have linkage specification 'extern "C"'}}
extern "C" struct A { int main(); }; // ok

namespace ns {
  extern "C" int main;  // expected-error {{'main' cannot have linkage specification 'extern "C"'}}
  extern "C" struct A {
    int main; // ok
  };

  extern "C" struct B {
    int main(); // ok
  };
}
