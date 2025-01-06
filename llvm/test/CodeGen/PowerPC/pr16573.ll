; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target triple = "powerpc64-unknown-linux-gnu"

define double @test() {
  %1 = fptrunc ppc_fp128 f0x800000000032D000818F2887B9295809 to double
  ret double %1
}

; CHECK: .quad 0x818f2887b9295809

