// RUN: not llvm-mc -triple aarch64-windows-msvc -filetype obj < %s 2>&1 | FileCheck %s

tail_call:
  adrp    x8, __imp_b
  ldr     x8, [x8, :lo12:__imp_b]
  .impcall        __imp_b
  br     x8

// CHECK: error: .impcall directives were used, but no existing .impcall section exists
