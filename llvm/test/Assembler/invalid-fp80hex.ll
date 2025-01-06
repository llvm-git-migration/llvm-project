; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Tests bug: 24640
; CHECK: expected '=' in global variable

@- f0xate potb8ed
