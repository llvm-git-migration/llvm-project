// This test examines whether BOLT can correctly update when
// dynamic relocation points to other entry points of the
// function.

# RUN: %clang %cflags -fPIC -pie %s -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s

    .text
    .type   chain, @function
chain:
    movq    $1, %rax
Lable:
    ret
    .size   chain, .-chain
    .type   _start, @function
    .global _start
_start:
    jmpq    *.Lfoo(%rip)
    ret
    .size   _start, .-_start
  .data
.Lfoo:
  .quad Lable

# CHECK-NOT: BOLT-ERROR
