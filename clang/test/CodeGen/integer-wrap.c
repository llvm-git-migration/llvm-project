// Check that -fsanitize=signed-integer-wrap instruments with -fwrapv
// RUN: %clang_cc1 -fwrapv -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=signed-integer-wrap | FileCheck %s --check-prefix=CHECK

// Check that -fsanitize=signed-integer-overflow doesn't instrument with -fwrapv
// RUN: %clang_cc1 -fwrapv -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=signed-integer-overflow | FileCheck %s --check-prefix=CHECKSIO

extern volatile int a, b, c;

// CHECK-LABEL: define void @test_add_overflow
void test_add_overflow(void) {
  // CHECK: [[ADD0:%.*]] = load {{.*}} i32
  // CHECK-NEXT: [[ADD1:%.*]] = load {{.*}} i32
  // CHECK-NEXT: {{%.*}} = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[ADD0]], i32 [[ADD1]])
  // CHECK: call void @__ubsan_handle_add_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_add_overflow
  a = b + c;
}

// CHECK-LABEL: define void @test_inc_overflow
void test_inc_overflow(void) {
  // This decays and gets handled by __ubsan_handle_add_overflow...
  // CHECK: [[INC0:%.*]] = load {{.*}} i32
  // CHECK-NEXT: call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[INC0]], i32 1)
  // CHECK: br {{.*}} %handler.add_overflow

  // CHECKSIO-NOT: br {{.*}} %handler.add_overflow
  ++a;
  a++;
}

// CHECK-LABEL: define void @test_sub_overflow
void test_sub_overflow(void) {
  // CHECK: [[SUB0:%.*]] = load {{.*}} i32
  // CHECK-NEXT: [[SUB1:%.*]] = load {{.*}} i32
  // CHECK-NEXT: call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 [[SUB0]], i32 [[SUB1]])
  // CHECK: call void @__ubsan_handle_sub_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_sub_overflow
  a = b - c;
}

// CHECK-LABEL: define void @test_mul_overflow
void test_mul_overflow(void) {
  // CHECK: [[MUL0:%.*]] = load {{.*}} i32
  // CHECK-NEXT: [[MUL1:%.*]] = load {{.*}} i32
  // CHECK-NEXT: call { i32, i1 } @llvm.smul.with.overflow.i32(i32 [[MUL0]], i32 [[MUL1]])
  // CHECK: call void @__ubsan_handle_mul_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_mul_overflow
  a = b * c;
}

// CHECK-LABEL: define void @test_div_overflow
void test_div_overflow(void) {
  // CHECK: [[DIV0:%.*]] = load {{.*}} i32
  // CHECK-NEXT: [[DIV1:%.*]] = load {{.*}} i32
  // CHECK-NEXT: [[DIV2:%.*]] = icmp ne i32 [[DIV0]], -2147483648
  // CHECK-NEXT: [[DIV3:%.*]] = icmp ne i32 [[DIV1]], -1
  // CHECK-NEXT: [[DIVOR:%or]] = or i1 [[DIV2]], [[DIV3]]
  // CHECK-NEXT: br {{.*}} %handler.divrem_overflow

  // -fsanitize=signed-integer-overflow still instruments division even with -fwrapv
  // CHECKSIO: br {{.*}} %handler.divrem_overflow
  a = b / c;
}
