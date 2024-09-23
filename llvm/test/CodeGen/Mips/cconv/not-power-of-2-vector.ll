; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips32 < %s | FileCheck %s --check-prefix=MIPS32

define void @call_v1i80() {
; MIPS32-LABEL: call_v1i80:
; MIPS32:       # %bb.0: # %Entry
; MIPS32-NEXT:    addiu $sp, $sp, -8
; MIPS32-NEXT:    .cfi_def_cfa_offset 8
; MIPS32-NEXT:    sw $ra, 4($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    .cfi_offset 31, -4
; MIPS32-NEXT:    addiu	$1, $zero, 4
; MIPS32-NEXT:    lw $1, 0($1)
; MIPS32-NEXT:    srl $2, $1, 16
; MIPS32-NEXT:    lw $3, 0($zero)
; MIPS32-NEXT:    sll $4, $3, 16
; MIPS32-NEXT:    or $5, $4, $2
; MIPS32-NEXT:    addiu	$2, $zero, 8
; MIPS32-NEXT:    lhu $2, 0($2)
; MIPS32-NEXT:    sll $1, $1, 16
; MIPS32-NEXT:    or $6, $2, $1
; MIPS32-NEXT:    addiu	$25, $zero, 0
; MIPS32-NEXT:    jalr $25
; MIPS32-NEXT:    srl $4, $3, 16
; MIPS32-NEXT:    lw $ra, 4($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    addiu $sp, $sp, 8
Entry:
  %0 = load <1 x i80>, ptr null, align 16
  call fastcc void null(<1 x i80> %0)
  ret void
}
