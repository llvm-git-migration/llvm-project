// RUN: %clang_cc1 -fsyntax-only -verify -pedantic-errors %s

namespace c {
  extern "C" void main(); // expected-error {{invalid linkage specification 'extern "C"'}}
}
extern "C" {
  int main(); // expected-error {{invalid linkage specification 'extern "C"'}}
}
extern "C" int main(); // expected-error {{invalid linkage specification 'extern "C"'}}
extern "C" struct A { int main(); }; // ok

namespace cpp {
  extern "C++" int main(); // expected-error {{invalid linkage specification 'extern "C++"'}}
}
extern "C++" {
  int main(); // expected-error {{invalid linkage specification 'extern "C++"'}}
}
extern "C++" int main(); // expected-error {{invalid linkage specification 'extern "C++"'}}
extern "C++" struct B { int main(); }; // ok

namespace ns {
  extern "C" struct A {
    int main; // ok
  };

  extern "C" struct B {
    int main(); // ok
  };
}
