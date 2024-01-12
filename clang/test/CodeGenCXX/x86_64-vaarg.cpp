// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef struct { struct {} a; } empty;

// CHECK-LABEL: define{{.*}} void @_Z17empty_record_testv()
empty empty_record_test(void) {
// CHECK: [[ADDR0:%[a-z._0-9]+]] = getelementptr inbounds %struct.__va_list_tag, ptr %arraydecay, i32 0, i32 2
// CHECK-NEXT: [[ADDR1:%[a-z._0-9]+]] = load ptr, ptr [[ADDR0]], align 8
// CHECK-NEXT: [[ADDR2:%[a-z._0-9]+]] = getelementptr i8, ptr [[ADDR1]], i32 0
  __builtin_va_list list;
  return __builtin_va_arg(list, empty);
}
