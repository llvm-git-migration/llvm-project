// RUN: not llvm-mc -triple aarch64-windows-msvc -filetype obj < %s 2>&1 | FileCheck %s

tail_call:
  adrp    x8, __imp_b
  ldr     x8, [x8, :lo12:__imp_b]
// CHECK: error: expected identifier in directive
  .impcall
  br     x8
// CHECK: error: unexpected token in directive
  .impcall        __imp_b x8
  br     x8

.section        .impcall,"yi"
.ascii  "Imp_Call_V1"
