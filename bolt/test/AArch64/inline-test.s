# This test checks that inline is properly handled by BOLT on aarch64.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux-gnu  %s -o %t.o
# RUN: %clang --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --inline-small-functions --print-inline  --print-only=_Z3barP1A  -debug-only=bolt-inliner %t.exe -o %t.bolt  | FileCheck %s
 
# CHECK: BOLT-INFO: inlined 0 calls at 1 call sites in 2 iteration(s). Change in binary size: 4 bytes.
# CHECK: Binary Function "_Z3barP1A" after inlining {
# CHECK-NOT: bl	_Z3fooP1A
# CHECK: ldr	x8, [x0]
# CHECK-NEXT: ldr	w0, [x8]
 
	.text
	.globl	_Z3fooP1A                       // -- Begin function _Z3fooP1A
	.p2align	2
	.type	_Z3fooP1A,@function
_Z3fooP1A:                              // @_Z3fooP1A
	.cfi_startproc
	ldr	x8, [x0]
	ldr	w0, [x8]
	ret
.Lfunc_end0:
	.size	_Z3fooP1A, .Lfunc_end0-_Z3fooP1A
	.cfi_endproc
	.globl	_Z3barP1A                       // -- Begin function _Z3barP1A
	.p2align	2
	.type	_Z3barP1A,@function
_Z3barP1A:                              // @_Z3barP1A
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	_Z3fooP1A
	mul	w0, w0, w0
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end1:
	.size	_Z3barP1A, .Lfunc_end1-_Z3barP1A
	.cfi_endproc
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
	mov	w0, wzr
	ret
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
	.section	".note.GNU-stack","",@progbits
	.addrsig