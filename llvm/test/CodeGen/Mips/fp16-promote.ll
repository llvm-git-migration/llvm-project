; RUN: llc -asm-verbose=false -mtriple=mipsel-linux-gnu -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-LIBCALL

; CHECK-LIBCALL-LABEL: test_fadd:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -40
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 40
; CHECK-LIBCALL-NEXT: sdc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: sw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: sw $17, 24($sp)
; CHECK-LIBCALL-NEXT: sw $16, 20($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 52, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 53, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -20
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $4
; CHECK-LIBCALL-NEXT: lhu $4, 0($5)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp0), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp0:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: lhu $4, 0($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp1), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp1:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: mov.s $f20, $f0
; CHECK-LIBCALL-NEXT: add.s $f12, $f0, $f20
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp2), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp2:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: sh $2, 0($17)
; CHECK-LIBCALL-NEXT: lw $16, 20($sp)
; CHECK-LIBCALL-NEXT: lw $17, 24($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: ldc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 40
define void @test_fadd(ptr %p, ptr %q) #0 {
  %a = load half, ptr %p, align 2
  %b = load half, ptr %q, align 2
  %r = fadd half %a, %b
  store half %r, ptr %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_fpext_float:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -24
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 24
; CHECK-LIBCALL-NEXT: sw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: addu $gp, $2, $25
; CHECK-LIBCALL-NEXT: lhu $4, 0($4)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($gp)
; CHECK-LIBCALL-NEXT: .reloc ($tmp3), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp3:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: nop
; CHECK-LIBCALL-NEXT: lw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 24
define float @test_fpext_float(ptr %p) {
  %a = load half, ptr %p, align 2
  %r = fpext half %a to float
  ret float %r
}

; CHECK-LIBCALL-LABEL: test_fpext_double:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -24
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 24
; CHECK-LIBCALL-NEXT: sw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: addu $gp, $2, $25
; CHECK-LIBCALL-NEXT: lhu $4, 0($4)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($gp)
; CHECK-LIBCALL-NEXT: .reloc ($tmp4), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp4:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: nop
; CHECK-LIBCALL-NEXT: cvt.d.s $f0, $f0
; CHECK-LIBCALL-NEXT: lw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 24
define double @test_fpext_double(ptr %p) {
  %a = load half, ptr %p, align 2
  %r = fpext half %a to double
  ret double %r
}

; CHECK-LIBCALL-LABEL: test_fptrunc_float:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -24
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 24
; CHECK-LIBCALL-NEXT: sw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -8
; CHECK-LIBCALL-NEXT: addu $gp, $2, $25
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($gp)
; CHECK-LIBCALL-NEXT: .reloc ($tmp5), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp5:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $16, $5
; CHECK-LIBCALL-NEXT: sh $2, 0($16)
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 24
define void @test_fptrunc_float(float %f, ptr %p) #0 {
  %a = fptrunc float %f to half
  store half %a, ptr %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_fptrunc_double:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -24
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 24
; CHECK-LIBCALL-NEXT: sw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -8
; CHECK-LIBCALL-NEXT: addu $gp, $2, $25
; CHECK-LIBCALL-NEXT: lw $25, %call16(__truncdfhf2)($gp)
; CHECK-LIBCALL-NEXT: .reloc ($tmp6), R_MIPS_JALR, __truncdfhf2
; CHECK-LIBCALL-NEXT: $tmp6:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $16, $6
; CHECK-LIBCALL-NEXT: sh $2, 0($16)
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 20($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 24
define void @test_fptrunc_double(double %d, ptr %p) #0 {
  %a = fptrunc double %d to half
  store half %a, ptr %p
  ret void
}

; CHECK-LIBCALL-LABEL: test_vec_fpext_float:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -40
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 40
; CHECK-LIBCALL-NEXT: sdc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: sw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: sw $18, 24($sp)
; CHECK-LIBCALL-NEXT: sw $17, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 52, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 53, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 18, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -20
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -24
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $5
; CHECK-LIBCALL-NEXT: move $18, $4
; CHECK-LIBCALL-NEXT: lhu $4, 4($5)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp7), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp7:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: lhu $4, 6($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp8), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp8:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: mov.s $f20, $f0
; CHECK-LIBCALL-NEXT: lhu $4, 2($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: swc1 $f0, 12($18)
; CHECK-LIBCALL-NEXT: .reloc ($tmp9), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp9:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: swc1 $f20, 8($18)
; CHECK-LIBCALL-NEXT: swc1 $f0, 4($18)
; CHECK-LIBCALL-NEXT: lhu $4, 0($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp10), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp10:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: nop
; CHECK-LIBCALL-NEXT: swc1 $f0, 0($18)
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $17, 20($sp)
; CHECK-LIBCALL-NEXT: lw $18, 24($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: ldc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 40
define <4 x float> @test_vec_fpext_float(ptr %p) #0 {
  %a = load <4 x half>, ptr %p, align 8
  %b = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %b
}

; This test is not robust against variations in instruction scheduling.
; See the discussion in http://reviews.llvm.org/D8804
; CHECK-LIBCALL-LABEL: test_vec_fpext_double:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -48
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 48
; CHECK-LIBCALL-NEXT: sdc1 $f22, 40($sp)
; CHECK-LIBCALL-NEXT: sdc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: sw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: sw $18, 24($sp)
; CHECK-LIBCALL-NEXT: sw $17, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 54, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 55, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 52, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 53, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -20
; CHECK-LIBCALL-NEXT: .cfi_offset 18, -24
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -28
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -32
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $5
; CHECK-LIBCALL-NEXT: move $18, $4
; CHECK-LIBCALL-NEXT: lhu $4, 2($5)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp11), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp11:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: lhu $4, 4($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp12), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp12:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: cvt.d.s $f20, $f0
; CHECK-LIBCALL-NEXT: lhu $4, 6($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp13), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp13:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: cvt.d.s $f22, $f0
; CHECK-LIBCALL-NEXT: cvt.d.s $f0, $f0
; CHECK-LIBCALL-NEXT: sdc1 $f0, 24($18)
; CHECK-LIBCALL-NEXT: sdc1 $f22, 16($18)
; CHECK-LIBCALL-NEXT: sdc1 $f20, 8($18)
; CHECK-LIBCALL-NEXT: lhu $4, 0($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp14), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp14:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: nop
; CHECK-LIBCALL-NEXT: cvt.d.s $f0, $f0
; CHECK-LIBCALL-NEXT: sdc1 $f0, 0($18)
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $17, 20($sp)
; CHECK-LIBCALL-NEXT: lw $18, 24($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: ldc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: ldc1 $f22, 40($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 48
define <4 x double> @test_vec_fpext_double(ptr %p) #0 {
  %a = load <4 x half>, ptr %p, align 8
  %b = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %b
}

; CHECK-LIBCALL-LABEL: test_vec_fptrunc_float:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -40
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 40
; CHECK-LIBCALL-NEXT: sw $ra, 36($sp)
; CHECK-LIBCALL-NEXT: sw $20, 32($sp)
; CHECK-LIBCALL-NEXT: sw $19, 28($sp)
; CHECK-LIBCALL-NEXT: sw $18, 24($sp)
; CHECK-LIBCALL-NEXT: sw $17, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 20, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 19, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 18, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -20
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -24
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $7
; CHECK-LIBCALL-NEXT: move $18, $5
; CHECK-LIBCALL-NEXT: move $19, $4
; CHECK-LIBCALL-NEXT: mtc1 $6, $f12
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp15), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp15:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: move $20, $2
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp16), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp16:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: mtc1 $17, $f12
; CHECK-LIBCALL-NEXT: mtc1 $18, $f12
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: lw $17, 56($sp)
; CHECK-LIBCALL-NEXT: sh $2, 6($17)
; CHECK-LIBCALL-NEXT: .reloc ($tmp17), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp17:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: sh $20, 4($17)
; CHECK-LIBCALL-NEXT: sh $2, 2($17)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp18), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp18:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: mtc1 $19, $f12
; CHECK-LIBCALL-NEXT: sh $2, 0($17)
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $17, 20($sp)
; CHECK-LIBCALL-NEXT: lw $18, 24($sp)
; CHECK-LIBCALL-NEXT: lw $19, 28($sp)
; CHECK-LIBCALL-NEXT: lw $20, 32($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 36($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 40
define void @test_vec_fptrunc_float(<4 x float> %a, ptr %p) #0 {
  %b = fptrunc <4 x float> %a to <4 x half>
  store <4 x half> %b, ptr %p, align 8
  ret void
}

; CHECK-LIBCALL-LABEL: test_vec_fptrunc_double:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -72
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 72
; CHECK-LIBCALL-NEXT: sw $ra, 68($sp)
; CHECK-LIBCALL-NEXT: sw $20, 64($sp)
; CHECK-LIBCALL-NEXT: sw $19, 60($sp)
; CHECK-LIBCALL-NEXT: sw $18, 56($sp)
; CHECK-LIBCALL-NEXT: sw $17, 52($sp)
; CHECK-LIBCALL-NEXT: sw $16, 48($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 20, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 19, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 18, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -20
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -24
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $5
; CHECK-LIBCALL-NEXT: move $18, $4
; CHECK-LIBCALL-NEXT: lw $1, 88($sp)
; CHECK-LIBCALL-NEXT: lw $2, 92($sp)
; CHECK-LIBCALL-NEXT: sw $2, 36($sp)
; CHECK-LIBCALL-NEXT: sw $1, 32($sp)
; CHECK-LIBCALL-NEXT: lw $1, 96($sp)
; CHECK-LIBCALL-NEXT: lw $2, 100($sp)
; CHECK-LIBCALL-NEXT: sw $2, 44($sp)
; CHECK-LIBCALL-NEXT: sw $1, 40($sp)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__truncdfhf2)($16)
; CHECK-LIBCALL-NEXT: ldc1 $f12, 32($sp)
; CHECK-LIBCALL-NEXT: sw $7, 28($sp)
; CHECK-LIBCALL-NEXT: sw $6, 24($sp)
; CHECK-LIBCALL-NEXT: .reloc ($tmp19), R_MIPS_JALR, __truncdfhf2
; CHECK-LIBCALL-NEXT: $tmp19:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: move $19, $2
; CHECK-LIBCALL-NEXT: lw $25, %call16(__truncdfhf2)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp20), R_MIPS_JALR, __truncdfhf2
; CHECK-LIBCALL-NEXT: $tmp20:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: ldc1 $f12, 40($sp)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__truncdfhf2)($16)
; CHECK-LIBCALL-NEXT: ldc1 $f12, 24($sp)
; CHECK-LIBCALL-NEXT: lw $20, 104($sp)
; CHECK-LIBCALL-NEXT: sh $2, 6($20)
; CHECK-LIBCALL-NEXT: .reloc ($tmp21), R_MIPS_JALR, __truncdfhf2
; CHECK-LIBCALL-NEXT: $tmp21:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: sh $19, 4($20)
; CHECK-LIBCALL-NEXT: sh $2, 2($20)
; CHECK-LIBCALL-NEXT: sw $17, 20($sp)
; CHECK-LIBCALL-NEXT: sw $18, 16($sp)
; CHECK-LIBCALL-NEXT: lw $25, %call16(__truncdfhf2)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp22), R_MIPS_JALR, __truncdfhf2
; CHECK-LIBCALL-NEXT: $tmp22:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: ldc1 $f12, 16($sp)
; CHECK-LIBCALL-NEXT: sh $2, 0($20)
; CHECK-LIBCALL-NEXT: lw $16, 48($sp)
; CHECK-LIBCALL-NEXT: lw $17, 52($sp)
; CHECK-LIBCALL-NEXT: lw $18, 56($sp)
; CHECK-LIBCALL-NEXT: lw $19, 60($sp)
; CHECK-LIBCALL-NEXT: lw $20, 64($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 68($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 72
define void @test_vec_fptrunc_double(<4 x double> %a, ptr %p) #0 {
  %b = fptrunc <4 x double> %a to <4 x half>
  store <4 x half> %b, ptr %p, align 8
  ret void
}

define half @test_fadd_fadd(half %a, half %b, half %c) {
; CHECK-LIBCALL-LABEL: test_fadd_fadd:
; CHECK-LIBCALL: lui $2, %hi(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $2, $2, %lo(_gp_disp)
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, -40
; CHECK-LIBCALL-NEXT: .cfi_def_cfa_offset 40
; CHECK-LIBCALL-NEXT: sdc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: sw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: sw $18, 24($sp)
; CHECK-LIBCALL-NEXT: sw $17, 20($sp)
; CHECK-LIBCALL-NEXT: sw $16, 16($sp)
; CHECK-LIBCALL-NEXT: .cfi_offset 52, -8
; CHECK-LIBCALL-NEXT: .cfi_offset 53, -4
; CHECK-LIBCALL-NEXT: .cfi_offset 31, -12
; CHECK-LIBCALL-NEXT: .cfi_offset 18, -16
; CHECK-LIBCALL-NEXT: .cfi_offset 17, -20
; CHECK-LIBCALL-NEXT: .cfi_offset 16, -24
; CHECK-LIBCALL-NEXT: addu $16, $2, $25
; CHECK-LIBCALL-NEXT: move $17, $6
; CHECK-LIBCALL-NEXT: move $18, $4
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: move $4, $5
; CHECK-LIBCALL-NEXT: .reloc ($tmp23), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp23:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: mov.s $f20, $f0
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp24), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp24:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $4, $18
; CHECK-LIBCALL-NEXT: add.s $f12, $f0, $f20
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp25), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp25:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $gp, $16
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp26), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp26:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $4, $2
; CHECK-LIBCALL-NEXT: mov.s $f20, $f0
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_h2f_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp27), R_MIPS_JALR, __gnu_h2f_ieee
; CHECK-LIBCALL-NEXT: $tmp27:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: move $4, $17
; CHECK-LIBCALL-NEXT: lw $25, %call16(__gnu_f2h_ieee)($16)
; CHECK-LIBCALL-NEXT: .reloc ($tmp28), R_MIPS_JALR, __gnu_f2h_ieee
; CHECK-LIBCALL-NEXT: $tmp28:
; CHECK-LIBCALL-NEXT: jalr $25
; CHECK-LIBCALL-NEXT: add.s $f12, $f20, $f0
; CHECK-LIBCALL-NEXT: lw $16, 16($sp)
; CHECK-LIBCALL-NEXT: lw $17, 20($sp)
; CHECK-LIBCALL-NEXT: lw $18, 24($sp)
; CHECK-LIBCALL-NEXT: lw $ra, 28($sp)
; CHECK-LIBCALL-NEXT: ldc1 $f20, 32($sp)
; CHECK-LIBCALL-NEXT: jr $ra
; CHECK-LIBCALL-NEXT: addiu $sp, $sp, 40
    %d = fadd half %a, %b
    %e = fadd half %d, %c
    ret half %e
}

