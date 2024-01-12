// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef struct { struct {} a; } empty;

// CHECK-LABEL: define{{.*}} void @_Z17empty_record_testv()
empty empty_record_test(void) {
// CHECK: [[RET:%[a-z]+]] = alloca %struct.empty, align 1
// CHECK: [[TMP:%[a-z]+]] = alloca %struct.empty, align 1
// CHECK: call void @llvm.memcpy{{.*}}(ptr align 1 [[RET]], ptr align 1 [[TMP]]
  __builtin_va_list list;
  return __builtin_va_arg(list, empty);
}
