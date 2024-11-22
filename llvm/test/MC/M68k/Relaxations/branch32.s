; RUN: llvm-mc -triple=m68k --mcpu=m68020 -motorola-integers -filetype=obj < %s \
; RUN:     | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: <RELAXED_32>:
RELAXED_32:
	; CHECK: bra  $ff
	bra	.LBB3_1
	.space 0x20000  ; Greater than u16::MAX.
.LBB3_1:
	add.l	#0, %d0
	rts