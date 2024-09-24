; NOTE: Do not autogenerate, we'll lose the important NOT- check otherwise.
;; Make sure we emit trap instructions after stack protector checks iff NoTrapAfterNoReturn is false.

; RUN: llc -mtriple=aarch64 -global-isel -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck -check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -global-isel -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck -check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -global-isel -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck -check-prefix=TRAP_UNREACHABLE %s

define void @test() nounwind ssp {
; NO_TRAP_UNREACHABLE-LABEL: test:
; NO_TRAP_UNREACHABLE:         bl __stack_chk_fail
; NO_TRAP_UNREACHABLE-NOT:     brk #0x1
;
; TRAP_UNREACHABLE-LABEL: test:
; TRAP_UNREACHABLE:         bl __stack_chk_fail
; TRAP_UNREACHABLE-NEXT:    brk #0x1

entry:
  %buf = alloca [8 x i8]
  %2 = call i32(ptr) @callee(ptr %buf) nounwind
  ret void
}

declare i32 @callee(ptr) nounwind
