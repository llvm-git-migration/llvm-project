// RUN: %clang_cc1 -triple riscv32 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV32I
// RUN: %clang_cc1 -triple riscv32 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV32IA
// RUN: %clang_cc1 -triple riscv64 -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV64I
// RUN: %clang_cc1 -triple riscv64 -target-feature +a -O1 -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=RV64IA

#include <stdatomic.h>
#include <stdint.h>

void test_i8_atomics(_Atomic(int8_t) * a, int8_t b) {
  // RV32I: load atomic i8, ptr %a seq_cst, align 1
  // RV32I: store atomic i8 %b, ptr %a seq_cst, align 1
  // RV32I: atomicrmw add ptr %a, i8 %b seq_cst, align 1
  // RV32IA: load atomic i8, ptr %a seq_cst, align 1
  // RV32IA: store atomic i8 %b, ptr %a seq_cst, align 1
  // RV32IA: atomicrmw add ptr %a, i8 %b seq_cst, align 1
  // RV64I: load atomic i8, ptr %a seq_cst, align 1
  // RV64I: store atomic i8 %b, ptr %a seq_cst, align 1
  // RV64I: atomicrmw add ptr %a, i8 %b seq_cst, align 1
  // RV64IA: load atomic i8, ptr %a seq_cst, align 1
  // RV64IA: store atomic i8 %b, ptr %a seq_cst, align 1
  // RV64IA: atomicrmw add ptr %a, i8 %b seq_cst, align 1
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i32_atomics(_Atomic(int32_t) * a, int32_t b) {
  // RV32I: load atomic i32, ptr %a seq_cst, align 4
  // RV32I: store atomic i32 %b, ptr %a seq_cst, align 4
  // RV32I: atomicrmw add ptr %a, i32 %b seq_cst, align 4
  // RV32IA: load atomic i32, ptr %a seq_cst, align 4
  // RV32IA: store atomic i32 %b, ptr %a seq_cst, align 4
  // RV32IA: atomicrmw add ptr %a, i32 %b seq_cst, align 4
  // RV64I: load atomic i32, ptr %a seq_cst, align 4
  // RV64I: store atomic i32 %b, ptr %a seq_cst, align 4
  // RV64I: atomicrmw add ptr %a, i32 %b seq_cst, align 4
  // RV64IA: load atomic i32, ptr %a seq_cst, align 4
  // RV64IA: store atomic i32 %b, ptr %a seq_cst, align 4
  // RV64IA: atomicrmw add ptr %a, i32 %b seq_cst, align 4
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}

void test_i64_atomics(_Atomic(int64_t) * a, int64_t b) {
  // RV32I: load atomic i64, ptr %a seq_cst, align 8
  // RV32I: store atomic i64 %b, ptr %a seq_cst, align 8
  // RV32I:  atomicrmw add ptr %a, i64 %b seq_cst, align 8
  // RV32IA: load atomic i64, ptr %a seq_cst, align 8
  // RV32IA: store atomic i64 %b, ptr %a seq_cst, align 8
  // RV32IA: atomicrmw add ptr %a, i64 %b seq_cst, align 8
  // RV64I: load atomic i64, ptr %a seq_cst, align 8
  // RV64I: store atomic i64 %b, ptr %a seq_cst, align 8
  // RV64I:  atomicrmw add ptr %a, i64 %b seq_cst, align 8
  // RV64IA: load atomic i64, ptr %a seq_cst, align 8
  // RV64IA: store atomic i64 %b, ptr %a seq_cst, align 8
  // RV64IA: atomicrmw add ptr %a, i64 %b seq_cst, align 8
  __c11_atomic_load(a, memory_order_seq_cst);
  __c11_atomic_store(a, b, memory_order_seq_cst);
  __c11_atomic_fetch_add(a, b, memory_order_seq_cst);
}
