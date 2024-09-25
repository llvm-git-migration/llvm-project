; NOTE: Do not autogenerate, we'd lose the .Lfunc_end0 -NEXT checks otherwise.
;; Make sure we emit trap instructions after stack protector checks iff NoTrapAfterNoReturn is false.

; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck --check-prefix=TRAP %s

; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck --check-prefix=TRAP %s

;; Make sure FastISel doesn't break anything.
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck --check-prefix=NOTRAP %s
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck --check-prefix=TRAP %s

define void @test() nounwind ssp {
; NOTRAP-LABEL: test
; NOTRAP:         callq __stack_chk_fail
; NOTRAP-NOT:     ud2
; NOTRAP-NEXT: .Lfunc_end0

; TRAP-LABEL: test
; TRAP:         callq __stack_chk_fail
; TRAP-NEXT:    ud2
; TRAP-NEXT: .Lfunc_end0

entry:
  %buf = alloca [8 x i8]
  %2 = call i32(ptr) @callee(ptr %buf) nounwind
  ret void
}

declare i32 @callee(ptr) nounwind
