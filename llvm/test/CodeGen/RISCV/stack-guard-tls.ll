; RUN: llc -mtriple=riscv64-unknown-elf < %s | \
; RUN: FileCheck %s

declare void @baz(ptr)

define void @foo(i64 %t) sspstrong {
  %vla = alloca i32, i64 %t, align 4
  call void @baz(ptr nonnull %vla)
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 2, !"stack-protector-guard", !"tls"}
!1 = !{i32 1, !"stack-protector-guard-reg", !"tp"}
!2 = !{i32 2, !"stack-protector-guard-offset", i32 500}

; CHECK: ld [[REG1:.*]], 500(tp)
; CHECK: call baz
; CHECK: ld [[REG2:.*]], 500(tp)
