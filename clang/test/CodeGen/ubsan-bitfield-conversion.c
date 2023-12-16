// RUN: %clang -fsanitize=implicit-integer-truncation -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-sign-change -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s
typedef struct _xx {
  int x1:3;
  char x2:2;
} xx, *pxx;

xx vxx;

// CHECK-LABEL: define{{.*}} void @foo1
void foo1(int x) {
  vxx.x1 = x;
  // CHECK: call void @__ubsan_handle_implicit_conversion
}

// CHECK: declare void @__ubsan_handle_implicit_conversion

// CHECK-LABEL: define{{.*}} void @foo2
void foo2(int x) {
  vxx.x2 = x;
  // CHECK: call void @__ubsan_handle_implicit_conversion
  // TODO: Ideally we should only emit once (emit is generated
  //       when evaluating RHS integer->char and when storing
  //       value in bitfield)
  // CHECK: call void @__ubsan_handle_implicit_conversion
}

