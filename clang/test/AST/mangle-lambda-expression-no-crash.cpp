// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

auto ICE = [](auto a) { return [=]<decltype(a) b>() { return 1; }; };
