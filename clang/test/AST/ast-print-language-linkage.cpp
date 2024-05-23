// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

// CHECK: extern "C" int printf(const char *, ...);
extern "C" int printf(const char *...);

// CHECK: extern "C++" {
// CHECK-NEXT:   int f(int);
// CHECK-NEXT:   int g(int);
// CHECK-NEXT: }
extern "C++" int f(int), g(int);

// CHECK: extern "C" {
// CHECK-NEXT:  void foo();
// CHECK-NEXT:  int x;
// CHECK-NEXT:  int y;
// CHECK-NEXT: }
extern "C" {
  void foo(void);
  int x, y;
}

// CHECK: extern "C" {
// CHECK-NEXT: }
extern "C" {}

// CHECK: extern "C++" ;
extern "C++";
