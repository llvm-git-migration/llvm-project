// RUN: %clang_analyze_cc1 -verify %s -fcxx-exceptions -fexceptions -analyzer-checker=core -analyzer-checker=alpha.deadcode.UnreachableCode

// expected-no-diagnostics

class BaseException {};

class DerivedException : public BaseException {};

void foo();

void f4() {
  try {
    foo();
  } catch (BaseException &b) {
  } catch (DerivedException &d) {
  }
}