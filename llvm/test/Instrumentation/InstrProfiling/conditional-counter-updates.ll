; RUN: opt < %s -S -passes=instrprof -conditional-counter-update | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"

; CHECK-LABEL: define void @foo
; CHECK-NEXT: %pgocount = load i8, ptr @__profc_foo, align 1
; CHECK-NEXT: %pgocount.ifnonzero = icmp ne i8 %pgocount, 0
; CHECK-NEXT: %pgocount.select = select i1 %pgocount.ifnonzero, i8 0, i8 %pgocount
; CHECK-NEXT: store i8 %pgocount.select, ptr @__profc_foo, align 1
define void @foo() {
  call void @llvm.instrprof.cover(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}
