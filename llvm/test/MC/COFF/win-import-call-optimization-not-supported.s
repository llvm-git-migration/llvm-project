// RUN: not llvm-mc -triple thumbv7a-windows-msvc < %s 2>&1 | FileCheck %s

tail_call:
  movw    r0, :lower16:__imp_a
  movt    r0, :upper16:__imp_a
  ldr     r0, [r0]
  pop.w   {r11, lr}
// CHECK: error: target doesn't have an import call section
  .impcall __imp_a
  bx      r0

.section        .impcall,"yi"
.ascii  "Imp_Call_V1"
