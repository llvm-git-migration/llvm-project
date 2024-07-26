; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld --lto-print-pipeline-passes -o %t.out.o %t.o 2>&1 | FileCheck %s

; CHECK: pipeline-passes: verify,{{.*}},verify

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.compiler.used = appending global [1 x ptr] [ptr @main], section "llvm.metadata"

define hidden void @main() {
  ret void
}
