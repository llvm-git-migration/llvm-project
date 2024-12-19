; RUN: llc -mtriple=aarch64-pc-windows-msvc -aarch64-win-import-call-optimization < %s | FileCheck %s --check-prefix=CHECK-ENABLED
; RUN: llc -mtriple=aarch64-pc-windows-msvc < %s | FileCheck %s --check-prefix=CHECK-DISABLED

; CHECK-DISABLED-NOT: .impcall
; CHECK-DISABLED-NOT: .section        .impcall

define dso_local void @normal_call() local_unnamed_addr {
entry:
  call void @a()
  call void @a()
  ret void
}
; CHECK-ENABLED-LABEL:  normal_call:
; CHECK-ENABLED:        adrp    [[ADRPREG:x[0-9]+]], __imp_a
; CHECK-ENABLED-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_a]
; CHECK-ENABLED-NEXT:   .impcall        __imp_a
; CHECK-ENABLED-NEXT:   blr     [[LDRREG]]
; CHECK-ENABLED-NEXT:   .impcall        __imp_a
; CHECK-ENABLED-NEXT:   blr     [[LDRREG]]

define dso_local void @tail_call() local_unnamed_addr {
entry:
  tail call void @b()
  ret void
}
; CHECK-ENABLED-LABEL:  tail_call:
; CHECK-ENABLED:        adrp    [[ADRPREG:x[0-9]+]], __imp_b
; CHECK-ENABLED-NEXT:   ldr     [[LDRREG:x[0-9]+]], [[[ADRPREG]], :lo12:__imp_b]
; CHECK-ENABLED-NEXT:   .impcall        __imp_b
; CHECK-ENABLED-NEXT:   br      [[LDRREG]]

declare dllimport void @a() local_unnamed_addr
declare dllimport void @b() local_unnamed_addr

; CHECK-ENABLED-LABEL  .section   .impcall,"yi"
; CHECK-ENABLED-NEXT   .asciz     "Imp_Call_V1"
