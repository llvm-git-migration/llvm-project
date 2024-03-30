// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s
// expected-no-diagnostics

enum {A, S, D, F};
int main() {
    using asdf = decltype(A);
    using enum asdf; // this line causes the crash
    return 0;
}
