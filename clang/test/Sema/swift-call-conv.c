// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only %s -verify
// RISC-V does not support swiftcall
// RUN: %clang_cc1 -triple riscv32-unknown-elf -fsyntax-only %s -verify

#if __has_feature(swiftcc)
// expected-no-diagnostics
#else
// expected-warning@+2 {{'__swiftcall__' calling convention is not supported for this target}}
#endif
void __attribute__((__swiftcall__)) f(void) {}
