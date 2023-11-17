// RUN: %clang_cc1 < %s -triple armv5e-none-linux-gnueabi -emit-llvm | FileCheck %s

enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
};

int *test_c11_atomic_fetch_add_int_ptr(_Atomic(int *) *p) {
  // CHECK-LABEL: define{{.*}} @test_c11_atomic_fetch_add_int_ptr
  // CHECK: store i32 12, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw add ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int *test_c11_atomic_fetch_sub_int_ptr(_Atomic(int *) *p) {
  // CHECK-LABEL: define{{.*}} @test_c11_atomic_fetch_sub_int_ptr
  // CHECK: store i32 20, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw sub ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int test_c11_atomic_fetch_add_int(_Atomic(int) *p) {
  // CHECK-LABEL: define{{.*}} @test_c11_atomic_fetch_add_int
  // CHECK: store i32 3, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw add ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int test_c11_atomic_fetch_sub_int(_Atomic(int) *p) {
  // CHECK-LABEL: define{{.*}} @test_c11_atomic_fetch_sub_int
  // CHECK: store i32 5, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw sub ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int *fp2a(int **p) {
  // CHECK-LABEL: define{{.*}} @fp2a
  // CHECK: store i32 4, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw sub ptr [[TMP0:%.*]], i32 [[TMP1]] monotonic, align 4
  // Note, the GNU builtins do not multiply by sizeof(T)!
  return __atomic_fetch_sub(p, 4, memory_order_relaxed);
}

int test_atomic_fetch_add(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_add
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw add ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_add(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_sub(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_sub
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw sub ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_sub(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_and(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_and
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw and ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_and(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_or(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_or
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw or ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_or(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_xor(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_xor
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw xor ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_xor(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_nand(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_fetch_nand
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: {{%[^ ]*}} = atomicrmw nand ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  return __atomic_fetch_nand(p, 55, memory_order_seq_cst);
}

int test_atomic_add_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_add_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw add ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: {{%[^ ]*}} = add i32 [[CALL]], [[TMP1]]
  return __atomic_add_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_sub_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_sub_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw sub ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: {{%[^ ]*}} = sub i32 [[CALL]], [[TMP1]]
  return __atomic_sub_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_and_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_and_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw and ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: {{%[^ ]*}} = and i32 [[CALL]], [[TMP1]]
  return __atomic_and_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_or_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_or_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw or ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: {{%[^ ]*}} = or i32 [[CALL]], [[TMP1]]
  return __atomic_or_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_xor_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_xor_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw xor ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: {{%[^ ]*}} = xor i32 [[CALL]], [[TMP1]]
  return __atomic_xor_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_nand_fetch(int *p) {
  // CHECK-LABEL: define{{.*}} @test_atomic_nand_fetch
  // CHECK: store i32 55, ptr [[ATOMICTMP:%[^ ]*]], align 4
  // CHECK: [[TMP1:%[^ ]*]] = load i32, ptr [[ATOMICTMP]], align 4
  // CHECK: [[CALL:%[^ ]*]] = atomicrmw nand ptr [[TMP0:%.*]], i32 [[TMP1]] seq_cst, align 4
  // CHECK: [[TMP2:%[^ ]*]] = and i32 [[CALL]], [[TMP1]]
  // CHECK: {{%[^ ]*}} = xor i32 [[TMP2]], -1
  return __atomic_nand_fetch(p, 55, memory_order_seq_cst);
}
