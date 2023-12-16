// RUN: %clang -fsanitize=implicit-bitfield-conversion -target x86_64-linux -S -emit-llvm -o - %s | FileCheck %s
//
typedef struct _xx {
  int x1:3;
  char x2:2;
} xx, *pxx;

xx vxx;

// CHECK-LABEL: define{{.*}} void @foo1
void foo1(int x) {
  vxx.x1 = x;
  // CHECK: call void @__ubsan_handle_implicit_bitfield_conversion
}

// CHECK-LABEL: define{{.*}} void @foo2
void foo2(int x) {
  vxx.x2 = x;
  // CHECK: call void @__ubsan_handle_implicit_bitfield_conversion
}

// CHECK-LABEL: define{{.*}} void @foo3
void foo3() {
  vxx.x1++;
  // CHECK: call void @__ubsan_handle_implicit_bitfield_conversion
}

// CHECK-LABEL: define{{.*}} void @foo4
void foo4(int x) {
  vxx.x1 += x;
  // CHECK: call void @__ubsan_handle_implicit_bitfield_conversion
}
