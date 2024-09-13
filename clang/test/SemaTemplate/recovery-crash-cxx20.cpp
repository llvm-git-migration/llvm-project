// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace GH49093 {
  class B {
  public:
    static int a() { return 0; } // expected-note {{member is declared here}}
    decltype(a< 0 >(0)) test;    // expected-error {{member 'a' used before its declaration}}
  };

  struct C {
      static int a() { return 0; } // expected-note {{member is declared here}}
      decltype(a < 0 > (0)) test;  // expected-error {{member 'a' used before its declaration}}
  };

  void test_is_bool(bool t) {}
  void test_is_bool(int t) {}

  int main() {
    B b;
    test_is_bool(b.test);

    C c;
    test_is_bool(c.test);
  }
}

namespace GH107047 {
  struct A {
    static constexpr auto test() { return 1; } // expected-note {{member is declared here}}
    static constexpr int s = test< 1 >();      // expected-error {{member 'test' used before its declaration}}
  };
}
