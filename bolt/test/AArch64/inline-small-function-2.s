## This test checks that inline is properly handled by BOLT on aarch64.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --inline-small-functions --print-inline  --print-only=test  \
# RUN: %t.exe -o %t.bolt | FileCheck %s

#CHECK: BOLT-INFO: inlined 0 calls at 1 call sites in 2 iteration(s). Change in binary size: 4 bytes.
#CHECK: Binary Function "test" after inlining {
#CHECK-NOT: bl	indirect
#CHECK: add	w0, w1, w0
#CHECK-NEXT: blr	x2
 
	.text
	.globl	indirect                       
	.type	indirect,@function
indirect:                               
	add	w0, w1, w0
	br	x2
.Lfunc_end0:
	.size	indirect, .Lfunc_end0-indirect

	.globl	test                           
	.type	test,@function
test:                                   
	stp	x29, x30, [sp, #-32]!          
	stp	x20, x19, [sp, #16]            
	mov	x29, sp
	mov	w19, w1
	mov	w20, w0
	bl	indirect
	add	w8, w19, w20
	cmp	w0, #0
	csinc	w0, w8, wzr, eq
	ldp	x20, x19, [sp, #16]             
	ldp	x29, x30, [sp], #32            
	ret
.Lfunc_end1:
	.size	test, .Lfunc_end1-test

	.globl	main                            
	.type	main,@function
main:                                   
	mov	w0, wzr
	ret
.Lfunc_end2:
	.size	main, .Lfunc_end2-main

	.section	".note.GNU-stack","",@progbits
	.addrsig
 