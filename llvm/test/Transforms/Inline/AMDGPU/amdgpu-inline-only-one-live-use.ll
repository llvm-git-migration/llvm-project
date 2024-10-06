; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-threshold=0 -debug-only=inline-cost %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-DEFAULT %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-threshold=0 -debug-only=inline-cost %s -amdgpu-inline-last-call-to-static-bonus=1024 -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-USER %s
; REQUIRES: asserts

; CHECK: Analyzing call of callee_not_only_one_live_use... (caller:caller)
; CHECK: Cost: -30
; CHECK: Threshold: 0
; CHECK: Analyzing call of callee_only_one_live_use... (caller:caller)
; CHECK-DEFAULT: Cost: -165030
; CHECK-USER: Cost: -1054
; CHECK: Threshold: 0

define internal void @callee_not_only_one_live_use() {
  ret void
}

define internal void @callee_only_one_live_use() {
  ret void
}

define void @caller() {
  call void @callee_not_only_one_live_use()
  call void @callee_not_only_one_live_use()
  call void @callee_only_one_live_use()
  ret void
}
