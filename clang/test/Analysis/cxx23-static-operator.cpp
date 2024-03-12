// RUN: %clang_analyze_cc1 -std=c++2b -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

template <typename T> void clang_analyzer_dump(T);

struct Adder {
  int data;
  static int operator()(int x, int y) {
    return x + y;
  }
};

void static_operator_call_inlines() {
  Adder s{10};
  clang_analyzer_dump(s(1, 2)); // expected-warning {{3 S32b}}
}
