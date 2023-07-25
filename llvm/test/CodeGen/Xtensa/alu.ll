; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s -check-prefix=XTENSA

; Register-immediate instructions

define i32 @addi(i32 %a) nounwind {
; XTENSA-LABEL: addi:
; XTENSA: addi a2, a2, 1
; XTENSA: ret
  %1 = add i32 %a, 1
  ret i32 %1
}

define i32 @addmi(i32 %a) nounwind {
; XTENSA-LABEL: addmi:
; XTENSA: addmi a2, a2, 32512
; XTENSA: ret
  %1 = add i32 %a, 32512
  ret i32 %1
}

define i32 @shrai(i32 %a) nounwind {
; XTENSA-LABEL: shrai:
; XTENSA: srai a2, a2, 4
; XTENSA: ret
  %1 = ashr i32 %a, 4
  ret i32 %1
}

define i32 @slli(i32 %a) nounwind {
; XTENSA-LABEL: slli:
; XTENSA: slli a2, a2, 4
; XTENSA: ret
  %1 = shl i32 %a, 4
  ret i32 %1
}

define i32 @srli(i32 %a) nounwind {
; XTENSA-LABEL: srli:
; XTENSA: srli a2, a2, 4
; XTENSA: ret
  %1 = lshr i32 %a, 4
  ret i32 %1
}

define i32 @movi(i32 %a) nounwind {
; XTENSA-LABEL: movi:
; XTENSA: movi	a8, 2047
; XTENSA: add	a2, a2, a8
; XTENSA: ret
  %1 = add i32 %a, 2047
  ret i32 %1
}

; Register-register instructions

define i32 @add(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: add:
; XTENSA: add a2, a2, a3
; XTENSA: ret
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @addx2(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: addx2:
; XTENSA: addx2 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 1
  %2 = add i32 %1, %b
  ret i32 %2
}

define i32 @addx4(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: addx4:
; XTENSA: addx4 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 2
  %2 = add i32 %1, %b
  ret i32 %2
}

define i32 @addx8(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: addx8:
; XTENSA: addx8 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 3
  %2 = add i32 %1, %b
  ret i32 %2
}

define i32 @sub(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: sub:
; XTENSA: sub a2, a2, a3
; XTENSA: ret
  %1 = sub i32 %a, %b
  ret i32 %1
}

define i32 @subx2(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: subx2:
; XTENSA: subx2 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 1
  %2 = sub i32 %1, %b
  ret i32 %2
}

define i32 @subx4(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: subx4:
; XTENSA: subx4 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 2
  %2 = sub i32 %1, %b
  ret i32 %2
}

define i32 @subx8(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: subx8:
; XTENSA: subx8 a2, a2, a3
; XTENSA: ret
  %1 = shl i32 %a, 3
  %2 = sub i32 %1, %b
  ret i32 %2
}

define i32 @xor(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: xor:
; XTENSA: xor a2, a2, a3
; XTENSA: ret
  %1 = xor i32 %a, %b
  ret i32 %1
}

define i32 @or(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: or:
; XTENSA: or a2, a2, a3
; XTENSA: ret
  %1 = or i32 %a, %b
  ret i32 %1
}

define i32 @and(i32 %a, i32 %b) nounwind {
; XTENSA-LABEL: and:
; XTENSA: and a2, a2, a3
; XTENSA: ret
  %1 = and i32 %a, %b
  ret i32 %1
}

define i32 @neg(i32 %a) nounwind {
; XTENSA-LABEL: neg:
; XTENSA: neg a2, a2
; XTENSA: ret
  %1 = sub i32 0, %a
  ret i32 %1
}

