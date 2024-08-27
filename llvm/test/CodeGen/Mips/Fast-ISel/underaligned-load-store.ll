; RUN: llc < %s -march mips -fast-isel -relocation-model pic | FileCheck %s -check-prefixes=MIPS

@var = external global i32, align 1

; FastISel should bail on the underaligned load and store.
define dso_local ccc i32 @__start() {
; MIPS:      lw  $1, %got(var)($1)
; MIPS-NEXT: lwl $2, 0($1)
; MIPS-NEXT: lwr $2, 3($1)
    %1 = load i32, ptr @var, align 1
; MIPS:      addiu $3, $zero, 42
; MIPS-NEXT: swl $3, 0($1)
; MIPS-NEXT: jr $ra
; MIPS-NEXT: swr $3, 3($1)
    store i32 42, ptr @var, align 1
    ret i32 %1
}
