; RUN: llc -O3 -mcpu=mips32r2 -mtriple=mipsel-linux-gnu < %s -o - \
; RUN:   | FileCheck %s --check-prefixes=MIPS32R2
; RUN: llc -O3 -mcpu=mips64r2 -march=mips64el  < %s \
; RUN:   | FileCheck %s --check-prefixes=MIPS64R2

define i32 @or_and_shl(i32 %a, i32 %b) {
; MIPS32R2-LABEL: or_and_shl:
; MIPS32R2:       # %bb.0: # %entry
; MIPS32R2-NEXT:    ins $4, $5, 31, 1
; MIPS32R2-NEXT:    jr $ra
; MIPS32R2-NEXT:    move $2, $4

entry:
  %shl = shl i32 %b, 31
  %and = and i32 %a, 2147483647
  %or = or i32 %and, %shl
  ret i32 %or
}

define i32 @or_shl_and(i32 %a, i32 %b) {
; MIPS32R2-LABEL: or_shl_and:
; MIPS32R2:       # %bb.0: # %entry
; MIPS32R2-NEXT:    ins $4, $5, 31, 1
; MIPS32R2-NEXT:    jr $ra
; MIPS32R2-NEXT:    move $2, $4

entry:
  %shl = shl i32 %b, 31
  %and = and i32 %a, 2147483647
  %or = or i32 %shl, %and
  ret i32 %or
}

define i64 @i64_or_and_shl(i64 %a, i64 %b) {
; MIPS64R2-LABEL: i64_or_and_shl:
; MIPS64R2:       # %bb.0: # %entry
; MIPS64R2-NEXT:    dins $4, $5, 31, 1
; MIPS64R2-NEXT:    jr $ra
; MIPS64R2-NEXT:    move $2, $4

entry:
  %shl = shl i64 %b, 31
  %and = and i64 %a, 2147483647
  %or = or i64 %and, %shl
  ret i64 %or
}

define i64 @i64_or_shl_and(i64 %a, i64 %b) {
; MIPS64R2-LABEL: i64_or_shl_and:
; MIPS64R2:       # %bb.0: # %entry
; MIPS64R2-NEXT:    dins $4, $5, 31, 1
; MIPS64R2-NEXT:    jr $ra
; MIPS64R2-NEXT:    move $2, $4

entry:
  %shl = shl i64 %b, 31
  %and = and i64 %a, 2147483647
  %or = or i64 %shl, %and
  ret i64 %or
}
