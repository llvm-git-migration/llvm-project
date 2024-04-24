// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: not clang-repl "int x = 10;" "int y=7; err;" "int y = 10;"
// RUN: cat %s | clang-repl | FileCheck %s
// RUN: cat %s | not clang-repl -Xcc -Xclang -Xcc -verify | FileCheck %s
BOOM! // expected-error {{intended to fail the -verify test}}
extern "C" int printf(const char *, ...);
int i = 42;
auto r1 = printf("i = %d\n", i);
// CHECK: i = 42
%quit
