	.text
	.syntax unified
	.eabi_attribute	67, "2.09"	@ Tag_conformance
	.eabi_attribute	6, 12	@ Tag_CPU_arch
	.eabi_attribute	7, 77	@ Tag_CPU_arch_profile
	.eabi_attribute	8, 0	@ Tag_ARM_ISA_use
	.eabi_attribute	9, 1	@ Tag_THUMB_ISA_use
	.eabi_attribute	34, 0	@ Tag_CPU_unaligned_access
	.eabi_attribute	17, 1	@ Tag_ABI_PCS_GOT_use
	.eabi_attribute	20, 1	@ Tag_ABI_FP_denormal
	.eabi_attribute	21, 1	@ Tag_ABI_FP_exceptions
	.eabi_attribute	23, 3	@ Tag_ABI_FP_number_model
	.eabi_attribute	24, 1	@ Tag_ABI_align_needed
	.eabi_attribute	25, 1	@ Tag_ABI_align_preserved
	.eabi_attribute	38, 1	@ Tag_ABI_FP_16bit_format
	.eabi_attribute	18, 4	@ Tag_ABI_PCS_wchar_t
	.eabi_attribute	26, 2	@ Tag_ABI_enum_size
	.eabi_attribute	14, 0	@ Tag_ABI_PCS_R9_use
	.file	"oggenc.ll"
	.globl	bark_noise_hybridmp             @ -- Begin function bark_noise_hybridmp
	.p2align	1
	.type	bark_noise_hybridmp,%function
	.code	16                              @ @bark_noise_hybridmp
	.thumb_func
bark_noise_hybridmp:
	.fnstart
@ %bb.0:                                @ %entry
	.save	{r4, r5, r6, r7, lr}
	push	{r4, r5, r6, r7, lr}
	.setfp	r7, sp, #12
	add	r7, sp, #12
	.pad	#124
	sub	sp, #124
	mov	r6, sp
	str	r3, [r6, #24]                   @ 4-byte Spill
	str	r1, [r6, #20]                   @ 4-byte Spill
	str	r0, [r6, #32]                   @ 4-byte Spill
	lsls	r0, r0, #2
	adds	r0, r0, #7
	movs	r1, #7
	bics	r0, r1
	mov	r1, sp
	subs	r1, r1, r0
	str	r1, [r6, #52]                   @ 4-byte Spill
	mov	sp, r1
	mov	r1, sp
	subs	r1, r1, r0
	str	r1, [r6, #48]                   @ 4-byte Spill
	mov	sp, r1
	mov	r1, sp
	subs	r3, r1, r0
	mov	sp, r3
	mov	r1, sp
	subs	r1, r1, r0
	str	r1, [r6, #44]                   @ 4-byte Spill
	mov	sp, r1
	mov	r1, sp
	subs	r0, r1, r0
	mov	sp, r0
	movs	r1, #0
	str	r3, [r6, #36]                   @ 4-byte Spill
	str	r1, [r3]
	str	r0, [r6, #40]                   @ 4-byte Spill
	str	r1, [r6, #120]                  @ 4-byte Spill
	str	r1, [r0]
	str	r2, [r6, #116]                  @ 4-byte Spill
	ldr	r0, [r2]
	ldr	r1, [r7, #8]
	str	r1, [r6, #112]                  @ 4-byte Spill
	bl	__aeabi_fadd
	mov	r4, r0
	movs	r0, #127
	str	r0, [r6, #108]                  @ 4-byte Spill
	lsls	r5, r0, #23
	mov	r0, r4
	mov	r1, r5
	bl	__aeabi_fcmplt
	cmp	r0, #0
	bne	.LBB0_2
@ %bb.1:                                @ %entry
	mov	r5, r4
.LBB0_2:                                @ %entry
	mov	r0, r5
	mov	r1, r5
	bl	__aeabi_fmul
	movs	r1, #63
	lsls	r1, r1, #24
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r1, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r1, r0
	ldr	r0, [r6, #52]                   @ 4-byte Reload
	str	r1, [r0]
	ldr	r0, [r6, #48]                   @ 4-byte Reload
	str	r1, [r6, #96]                   @ 4-byte Spill
	str	r1, [r0]
	mov	r0, r5
	mov	r1, r4
	bl	__aeabi_fmul
	ldr	r1, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r1, r0
	ldr	r0, [r6, #44]                   @ 4-byte Reload
	str	r1, [r6, #104]                  @ 4-byte Spill
	str	r1, [r0]
	ldr	r0, [r6, #32]                   @ 4-byte Reload
	cmp	r0, #2
	blt	.LBB0_7
@ %bb.3:                                @ %for.body.lr.ph
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r4, r0, #23
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	adds	r1, r0, #4
	ldr	r0, [r6, #52]                   @ 4-byte Reload
	adds	r5, r0, #4
	ldr	r0, [r6, #48]                   @ 4-byte Reload
	adds	r0, r0, #4
	str	r0, [r6, #92]                   @ 4-byte Spill
	ldr	r0, [r6, #36]                   @ 4-byte Reload
	adds	r0, r0, #4
	str	r0, [r6, #88]                   @ 4-byte Spill
	ldr	r0, [r6, #44]                   @ 4-byte Reload
	adds	r0, r0, #4
	str	r0, [r6, #84]                   @ 4-byte Spill
	ldr	r0, [r6, #40]                   @ 4-byte Reload
	adds	r0, r0, #4
	str	r0, [r6, #80]                   @ 4-byte Spill
	ldr	r0, [r6, #32]                   @ 4-byte Reload
	subs	r3, r0, #1
	movs	r2, #0
	mov	r0, r4
	str	r2, [r6, #76]                   @ 4-byte Spill
	str	r2, [r6, #72]                   @ 4-byte Spill
	ldr	r2, [r6, #96]                   @ 4-byte Reload
	str	r2, [r6, #100]                  @ 4-byte Spill
	str	r4, [r6, #28]                   @ 4-byte Spill
	b	.LBB0_5
.LBB0_4:                                @ %for.body
                                        @   in Loop: Header=BB0_5 Depth=1
	mov	r0, r4
	mov	r1, r4
	bl	__aeabi_fmul
	mov	r5, r0
	mov	r0, r4
	mov	r1, r5
	str	r5, [r6, #56]                   @ 4-byte Spill
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #104]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #104]                  @ 4-byte Spill
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r5, r0
	mov	r0, r4
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #76]                   @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r1, r0
	ldr	r0, [r6, #84]                   @ 4-byte Reload
	ldr	r2, [r6, #104]                  @ 4-byte Reload
	stm	r0!, {r2}
	str	r0, [r6, #84]                   @ 4-byte Spill
	ldr	r0, [r6, #80]                   @ 4-byte Reload
	str	r1, [r6, #76]                   @ 4-byte Spill
	stm	r0!, {r1}
	str	r0, [r6, #80]                   @ 4-byte Spill
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	ldr	r1, [r6, #56]                   @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #100]                  @ 4-byte Spill
	ldr	r0, [r6, #96]                   @ 4-byte Reload
	mov	r1, r5
	bl	__aeabi_fadd
	mov	r4, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #72]                   @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r1, r0
	ldr	r5, [r6, #60]                   @ 4-byte Reload
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	stm	r5!, {r0}
	str	r4, [r6, #96]                   @ 4-byte Spill
	ldr	r0, [r6, #92]                   @ 4-byte Reload
	stm	r0!, {r4}
	str	r0, [r6, #92]                   @ 4-byte Spill
	ldr	r0, [r6, #88]                   @ 4-byte Reload
	str	r1, [r6, #72]                   @ 4-byte Spill
	stm	r0!, {r1}
	str	r0, [r6, #88]                   @ 4-byte Spill
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	ldr	r4, [r6, #28]                   @ 4-byte Reload
	mov	r1, r4
	bl	__aeabi_fadd
	ldr	r2, [r6, #64]                   @ 4-byte Reload
	subs	r3, r2, #1
	ldr	r1, [r6, #68]                   @ 4-byte Reload
	beq	.LBB0_7
.LBB0_5:                                @ %for.body
                                        @ =>This Inner Loop Header: Depth=1
	str	r5, [r6, #60]                   @ 4-byte Spill
	str	r3, [r6, #64]                   @ 4-byte Spill
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldm	r1!, {r0}
	str	r1, [r6, #68]                   @ 4-byte Spill
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r5, r0
	mov	r1, r4
	bl	__aeabi_fcmplt
	cmp	r0, #0
	bne	.LBB0_4
@ %bb.6:                                @ %for.body
                                        @   in Loop: Header=BB0_5 Depth=1
	mov	r4, r5
	b	.LBB0_4
.LBB0_7:                                @ %for.cond43.preheader
	ldr	r4, [r6, #20]                   @ 4-byte Reload
	ldr	r1, [r4]
	asrs	r2, r1, #16
	bmi	.LBB0_11
@ %bb.8:
                                        @ implicit-def: $r0
                                        @ kill: killed $r0
                                        @ implicit-def: $r0
                                        @ kill: killed $r0
                                        @ implicit-def: $r0
                                        @ kill: killed $r0
	ldr	r3, [r6, #120]                  @ 4-byte Reload
	mov	r5, r3
	uxth	r0, r1
	ldr	r2, [r6, #32]                   @ 4-byte Reload
	cmp	r0, r2
	bge	.LBB0_9
	b	.LBB0_16
.LBB0_9:                                @ %for.cond136.preheader
	ldr	r0, [r7, #12]
	str	r0, [r6, #8]                    @ 4-byte Spill
	cmp	r2, r5
	ble	.LBB0_10
	b	.LBB0_21
.LBB0_10:                               @ %for.cond136.preheader
	b	.LBB0_25
.LBB0_11:                               @ %if.end48.preheader
	adds	r0, r4, #4
	movs	r5, #0
	ldr	r4, [r6, #24]                   @ 4-byte Reload
	mov	r3, r5
	str	r5, [r6, #72]                   @ 4-byte Spill
	b	.LBB0_13
.LBB0_12:                               @ %if.end48
                                        @   in Loop: Header=BB0_13 Depth=1
	mov	r0, r5
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	ldr	r4, [r6, #88]                   @ 4-byte Reload
	stm	r4!, {r0}
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r3, r0
	ldr	r5, [r6, #96]                   @ 4-byte Reload
	adds	r5, r5, #1
	ldr	r0, [r6, #92]                   @ 4-byte Reload
	ldm	r0!, {r1}
	asrs	r2, r1, #16
	bpl	.LBB0_15
.LBB0_13:                               @ %if.end48
                                        @ =>This Inner Loop Header: Depth=1
	str	r3, [r6, #120]                  @ 4-byte Spill
	str	r4, [r6, #88]                   @ 4-byte Spill
	str	r0, [r6, #92]                   @ 4-byte Spill
	str	r5, [r6, #96]                   @ 4-byte Spill
	lsls	r0, r1, #16
	lsrs	r5, r0, #14
	ldr	r1, [r6, #36]                   @ 4-byte Reload
	ldr	r0, [r1, r5]
	lsls	r4, r2, #2
	subs	r1, r1, r4
	ldr	r1, [r1]
	bl	__aeabi_fadd
	str	r0, [r6, #100]                  @ 4-byte Spill
	ldr	r1, [r6, #44]                   @ 4-byte Reload
	str	r5, [r6, #84]                   @ 4-byte Spill
	ldr	r0, [r1, r5]
	str	r4, [r6, #76]                   @ 4-byte Spill
	subs	r1, r1, r4
	ldr	r1, [r1]
	bl	__aeabi_fadd
	mov	r1, r0
	str	r0, [r6, #104]                  @ 4-byte Spill
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	bl	__aeabi_fmul
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r1, [r6, #48]                   @ 4-byte Reload
	ldr	r0, [r1, r5]
	subs	r1, r1, r4
	ldr	r1, [r1]
	bl	__aeabi_fsub
	mov	r4, r0
	str	r0, [r6, #80]                   @ 4-byte Spill
	ldr	r1, [r6, #40]                   @ 4-byte Reload
	ldr	r0, [r1, r5]
	ldr	r5, [r6, #76]                   @ 4-byte Reload
	subs	r1, r1, r5
	ldr	r1, [r1]
	bl	__aeabi_fsub
	mov	r1, r0
	mov	r0, r4
	mov	r4, r1
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fsub
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r1, [r6, #52]                   @ 4-byte Reload
	ldr	r0, [r6, #84]                   @ 4-byte Reload
	ldr	r0, [r1, r0]
	subs	r1, r1, r5
	ldr	r1, [r1]
	bl	__aeabi_fadd
	str	r0, [r6, #84]                   @ 4-byte Spill
	mov	r1, r4
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r5, [r6, #80]                   @ 4-byte Reload
	mov	r0, r5
	ldr	r1, [r6, #104]                  @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	str	r1, [r6, #104]                  @ 4-byte Spill
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #76]                   @ 4-byte Spill
	ldr	r0, [r6, #84]                   @ 4-byte Reload
	ldr	r1, [r6, #100]                  @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r4, r0
	mov	r0, r5
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #76]                   @ 4-byte Reload
	str	r1, [r6, #100]                  @ 4-byte Spill
	bl	__aeabi_fdiv
	mov	r4, r0
	ldr	r5, [r6, #72]                   @ 4-byte Reload
	mov	r1, r5
	bl	__aeabi_fcmplt
	cmp	r0, #0
	beq	.LBB0_14
	b	.LBB0_12
.LBB0_14:                               @ %if.end48
                                        @   in Loop: Header=BB0_13 Depth=1
	mov	r5, r4
	b	.LBB0_12
.LBB0_15:
	ldr	r4, [r6, #20]                   @ 4-byte Reload
	uxth	r0, r1
	ldr	r2, [r6, #32]                   @ 4-byte Reload
	cmp	r0, r2
	blt	.LBB0_16
	b	.LBB0_9
.LBB0_16:                               @ %if.end98.preheader
	lsls	r2, r5, #2
	str	r3, [r6, #120]                  @ 4-byte Spill
	ldr	r3, [r6, #24]                   @ 4-byte Reload
	adds	r3, r3, r2
	str	r3, [r6, #92]                   @ 4-byte Spill
	adds	r2, r2, r4
	adds	r3, r2, #4
	b	.LBB0_18
.LBB0_17:                               @ %if.end98
                                        @   in Loop: Header=BB0_18 Depth=1
	mov	r0, r4
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	ldr	r1, [r6, #92]                   @ 4-byte Reload
	stm	r1!, {r0}
	str	r1, [r6, #92]                   @ 4-byte Spill
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #120]                  @ 4-byte Spill
	ldr	r5, [r6, #96]                   @ 4-byte Reload
	adds	r5, r5, #1
	ldr	r3, [r6, #88]                   @ 4-byte Reload
	ldm	r3!, {r1}
	uxth	r0, r1
	ldr	r2, [r6, #32]                   @ 4-byte Reload
	cmp	r0, r2
	bge	.LBB0_20
.LBB0_18:                               @ %if.end98
                                        @ =>This Inner Loop Header: Depth=1
	str	r3, [r6, #88]                   @ 4-byte Spill
	str	r5, [r6, #96]                   @ 4-byte Spill
	lsls	r0, r0, #2
	str	r0, [r6, #104]                  @ 4-byte Spill
	ldr	r2, [r6, #36]                   @ 4-byte Reload
	ldr	r0, [r2, r0]
	asrs	r1, r1, #16
	lsls	r4, r1, #2
	ldr	r1, [r2, r4]
	bl	__aeabi_fsub
	mov	r5, r0
	str	r0, [r6, #84]                   @ 4-byte Spill
	ldr	r1, [r6, #44]                   @ 4-byte Reload
	ldr	r0, [r6, #104]                  @ 4-byte Reload
	ldr	r0, [r1, r0]
	ldr	r1, [r1, r4]
	bl	__aeabi_fsub
	mov	r1, r0
	str	r0, [r6, #80]                   @ 4-byte Spill
	mov	r0, r5
	bl	__aeabi_fmul
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r1, [r6, #48]                   @ 4-byte Reload
	ldr	r5, [r6, #104]                  @ 4-byte Reload
	ldr	r0, [r1, r5]
	ldr	r1, [r1, r4]
	bl	__aeabi_fsub
	str	r0, [r6, #100]                  @ 4-byte Spill
	ldr	r1, [r6, #40]                   @ 4-byte Reload
	ldr	r0, [r1, r5]
	ldr	r1, [r1, r4]
	bl	__aeabi_fsub
	mov	r5, r0
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fsub
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r1, [r6, #52]                   @ 4-byte Reload
	ldr	r0, [r6, #104]                  @ 4-byte Reload
	ldr	r0, [r1, r0]
	ldr	r1, [r1, r4]
	bl	__aeabi_fsub
	str	r0, [r6, #76]                   @ 4-byte Spill
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r5, [r6, #100]                  @ 4-byte Reload
	mov	r0, r5
	ldr	r1, [r6, #80]                   @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	str	r1, [r6, #104]                  @ 4-byte Spill
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #80]                   @ 4-byte Spill
	ldr	r0, [r6, #76]                   @ 4-byte Reload
	ldr	r1, [r6, #84]                   @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r4, r0
	mov	r0, r5
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #80]                   @ 4-byte Reload
	str	r1, [r6, #100]                  @ 4-byte Spill
	bl	__aeabi_fdiv
	mov	r5, r0
	movs	r4, #0
	mov	r1, r4
	bl	__aeabi_fcmplt
	cmp	r0, #0
	bne	.LBB0_17
@ %bb.19:                               @ %if.end98
                                        @   in Loop: Header=BB0_18 Depth=1
	mov	r4, r5
	b	.LBB0_17
.LBB0_20:
	ldr	r3, [r6, #120]                  @ 4-byte Reload
	ldr	r0, [r7, #12]
	str	r0, [r6, #8]                    @ 4-byte Spill
	cmp	r2, r5
	ble	.LBB0_25
.LBB0_21:                               @ %for.body139.lr.ph
	ldr	r0, [r6, #32]                   @ 4-byte Reload
	subs	r2, r0, r5
	lsls	r0, r5, #2
	ldr	r1, [r6, #24]                   @ 4-byte Reload
	adds	r5, r1, r0
	b	.LBB0_23
.LBB0_22:                               @ %for.body139
                                        @   in Loop: Header=BB0_23 Depth=1
	mov	r0, r4
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	stm	r5!, {r0}
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r3, r0
	ldr	r0, [r6, #96]                   @ 4-byte Reload
	subs	r2, r0, #1
	beq	.LBB0_25
.LBB0_23:                               @ %for.body139
                                        @ =>This Inner Loop Header: Depth=1
	str	r2, [r6, #96]                   @ 4-byte Spill
	ldr	r0, [r6, #104]                  @ 4-byte Reload
	str	r3, [r6, #120]                  @ 4-byte Spill
	mov	r1, r3
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	ldr	r1, [r6, #100]                  @ 4-byte Reload
	bl	__aeabi_fdiv
	movs	r4, #0
	str	r0, [r6, #92]                   @ 4-byte Spill
	mov	r1, r4
	bl	__aeabi_fcmplt
	cmp	r0, #0
	bne	.LBB0_22
@ %bb.24:                               @ %for.body139
                                        @   in Loop: Header=BB0_23 Depth=1
	ldr	r4, [r6, #92]                   @ 4-byte Reload
	b	.LBB0_22
.LBB0_25:                               @ %for.end152
	ldr	r1, [r6, #8]                    @ 4-byte Reload
	cmp	r1, #1
	ldr	r3, [r6, #32]                   @ 4-byte Reload
	bge	.LBB0_26
	b	.LBB0_46
.LBB0_26:                               @ %for.cond157.preheader
	lsrs	r0, r1, #31
	adds	r0, r1, r0
	asrs	r4, r0, #1
	subs	r0, r4, r1
	movs	r2, #0
	cmp	r0, #0
	mov	r0, r2
	bmi	.LBB0_30
@ %bb.27:                               @ %for.cond209.preheader
	adds	r5, r0, r4
	cmp	r5, r3
	bge	.LBB0_28
	b	.LBB0_35
.LBB0_28:                               @ %for.cond256.preheader
	cmp	r0, r3
	bge	.LBB0_29
	b	.LBB0_40
.LBB0_29:                               @ %for.cond256.preheader
	b	.LBB0_46
.LBB0_30:                               @ %if.end164.lr.ph
	subs	r0, r1, r4
	str	r0, [r6, #80]                   @ 4-byte Spill
	str	r4, [r6, #4]                    @ 4-byte Spill
	lsls	r0, r4, #2
	ldr	r2, [r6, #52]                   @ 4-byte Reload
	adds	r3, r2, r0
	str	r3, [r6, #56]                   @ 4-byte Spill
	ldr	r2, [r6, #48]                   @ 4-byte Reload
	adds	r4, r2, r0
	str	r4, [r6, #28]                   @ 4-byte Spill
	ldr	r4, [r6, #36]                   @ 4-byte Reload
	adds	r5, r4, r0
	str	r5, [r6, #20]                   @ 4-byte Spill
	ldr	r5, [r6, #44]                   @ 4-byte Reload
	adds	r2, r5, r0
	str	r2, [r6, #16]                   @ 4-byte Spill
	ldr	r2, [r6, #40]                   @ 4-byte Reload
	adds	r2, r2, r0
	str	r2, [r6, #12]                   @ 4-byte Spill
	lsls	r1, r1, #2
	subs	r0, r1, r0
	ldr	r1, [r6, #52]                   @ 4-byte Reload
	adds	r1, r1, r0
	ldr	r2, [r6, #48]                   @ 4-byte Reload
	adds	r3, r2, r0
	adds	r4, r4, r0
	adds	r5, r5, r0
	ldr	r2, [r6, #40]                   @ 4-byte Reload
	adds	r0, r2, r0
	str	r0, [r6, #92]                   @ 4-byte Spill
	movs	r2, #0
	str	r2, [r6, #96]                   @ 4-byte Spill
	b	.LBB0_32
.LBB0_31:                               @ %for.inc205
                                        @   in Loop: Header=BB0_32 Depth=1
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #120]                  @ 4-byte Spill
	ldr	r0, [r6, #80]                   @ 4-byte Reload
	subs	r0, r0, #1
	adds	r5, r5, #4
	str	r5, [r6, #96]                   @ 4-byte Spill
	ldr	r1, [r6, #88]                   @ 4-byte Reload
	subs	r1, r1, #4
	ldr	r3, [r6, #76]                   @ 4-byte Reload
	subs	r3, r3, #4
	ldr	r4, [r6, #72]                   @ 4-byte Reload
	subs	r4, r4, #4
	ldr	r5, [r6, #68]                   @ 4-byte Reload
	subs	r5, r5, #4
	ldr	r2, [r6, #92]                   @ 4-byte Reload
	subs	r2, r2, #4
	str	r2, [r6, #92]                   @ 4-byte Spill
	ldr	r2, [r6, #120]                  @ 4-byte Reload
	str	r0, [r6, #80]                   @ 4-byte Spill
	cmp	r0, #0
	beq	.LBB0_34
.LBB0_32:                               @ %if.end164
                                        @ =>This Inner Loop Header: Depth=1
	str	r1, [r6, #88]                   @ 4-byte Spill
	str	r2, [r6, #120]                  @ 4-byte Spill
	ldr	r0, [r6, #28]                   @ 4-byte Reload
	ldr	r1, [r6, #96]                   @ 4-byte Reload
	ldr	r0, [r0, r1]
	str	r3, [r6, #76]                   @ 4-byte Spill
	ldr	r1, [r3]
	bl	__aeabi_fsub
	str	r0, [r6, #84]                   @ 4-byte Spill
	ldr	r0, [r6, #20]                   @ 4-byte Reload
	ldr	r1, [r6, #96]                   @ 4-byte Reload
	ldr	r0, [r0, r1]
	str	r4, [r6, #72]                   @ 4-byte Spill
	ldr	r1, [r4]
	bl	__aeabi_fadd
	str	r0, [r6, #100]                  @ 4-byte Spill
	ldr	r0, [r6, #16]                   @ 4-byte Reload
	ldr	r2, [r6, #96]                   @ 4-byte Reload
	ldr	r0, [r0, r2]
	str	r5, [r6, #68]                   @ 4-byte Spill
	ldr	r1, [r5]
	mov	r5, r2
	bl	__aeabi_fadd
	mov	r4, r0
	str	r0, [r6, #104]                  @ 4-byte Spill
	ldr	r0, [r6, #12]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #92]                   @ 4-byte Reload
	ldr	r1, [r1]
	bl	__aeabi_fsub
	mov	r5, r0
	str	r0, [r6, #60]                   @ 4-byte Spill
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	mov	r1, r4
	bl	__aeabi_fmul
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r4, [r6, #84]                   @ 4-byte Reload
	mov	r0, r4
	mov	r1, r5
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fsub
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r0, [r6, #56]                   @ 4-byte Reload
	ldr	r5, [r6, #96]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #88]                   @ 4-byte Reload
	ldr	r1, [r1]
	bl	__aeabi_fadd
	str	r0, [r6, #64]                   @ 4-byte Spill
	ldr	r1, [r6, #60]                   @ 4-byte Reload
	bl	__aeabi_fmul
	str	r0, [r6, #60]                   @ 4-byte Spill
	mov	r0, r4
	ldr	r1, [r6, #104]                  @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #60]                   @ 4-byte Reload
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	str	r1, [r6, #104]                  @ 4-byte Spill
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #60]                   @ 4-byte Spill
	ldr	r0, [r6, #64]                   @ 4-byte Reload
	ldr	r1, [r6, #100]                  @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r0, [r6, #84]                   @ 4-byte Reload
	mov	r1, r0
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #60]                   @ 4-byte Reload
	str	r1, [r6, #100]                  @ 4-byte Spill
	bl	__aeabi_fdiv
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	mov	r4, r0
	ldr	r0, [r6, #24]                   @ 4-byte Reload
	ldr	r1, [r0, r5]
	mov	r0, r4
	bl	__aeabi_fcmplt
	cmp	r0, #0
	bne	.LBB0_33
	b	.LBB0_31
.LBB0_33:                               @ %if.then201
                                        @   in Loop: Header=BB0_32 Depth=1
	ldr	r0, [r6, #24]                   @ 4-byte Reload
	str	r4, [r0, r5]
	b	.LBB0_31
.LBB0_34:                               @ %for.cond157.for.cond209.preheader_crit_edge
	ldr	r4, [r6, #4]                    @ 4-byte Reload
	ldr	r1, [r6, #8]                    @ 4-byte Reload
	subs	r0, r1, r4
	ldr	r3, [r6, #32]                   @ 4-byte Reload
	adds	r5, r0, r4
	cmp	r5, r3
	blt	.LBB0_35
	b	.LBB0_28
.LBB0_35:                               @ %if.end216.lr.ph
	str	r5, [r6, #28]                   @ 4-byte Spill
	lsls	r1, r0, #2
	str	r2, [r6, #120]                  @ 4-byte Spill
	ldr	r2, [r6, #24]                   @ 4-byte Reload
	adds	r2, r2, r1
	str	r2, [r6, #80]                   @ 4-byte Spill
	adds	r0, r0, r4
	str	r0, [r6, #116]                  @ 4-byte Spill
	subs	r2, r0, r3
	str	r2, [r6, #96]                   @ 4-byte Spill
	str	r4, [r6, #4]                    @ 4-byte Spill
	lsls	r2, r4, #2
	adds	r1, r1, r2
	ldr	r0, [r6, #52]                   @ 4-byte Reload
	adds	r3, r0, r1
	str	r3, [r6, #76]                   @ 4-byte Spill
	ldr	r3, [r6, #48]                   @ 4-byte Reload
	adds	r4, r3, r1
	str	r4, [r6, #72]                   @ 4-byte Spill
	ldr	r4, [r6, #36]                   @ 4-byte Reload
	adds	r5, r4, r1
	str	r5, [r6, #68]                   @ 4-byte Spill
	ldr	r5, [r6, #44]                   @ 4-byte Reload
	adds	r0, r5, r1
	str	r0, [r6, #64]                   @ 4-byte Spill
	ldr	r2, [r6, #40]                   @ 4-byte Reload
	adds	r0, r2, r1
	str	r0, [r6, #60]                   @ 4-byte Spill
	ldr	r0, [r6, #8]                    @ 4-byte Reload
	ldr	r1, [r6, #116]                  @ 4-byte Reload
	subs	r0, r1, r0
	lsls	r0, r0, #2
	ldr	r1, [r6, #52]                   @ 4-byte Reload
	adds	r1, r1, r0
	str	r1, [r6, #56]                   @ 4-byte Spill
	adds	r1, r3, r0
	str	r1, [r6, #52]                   @ 4-byte Spill
	adds	r1, r4, r0
	str	r1, [r6, #48]                   @ 4-byte Spill
	adds	r1, r5, r0
	str	r1, [r6, #44]                   @ 4-byte Spill
	adds	r0, r2, r0
	str	r0, [r6, #40]                   @ 4-byte Spill
	movs	r5, #0
	b	.LBB0_37
.LBB0_36:                               @ %for.inc252
                                        @   in Loop: Header=BB0_37 Depth=1
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #120]                  @ 4-byte Spill
	ldr	r1, [r6, #96]                   @ 4-byte Reload
	adds	r0, r1, #1
	adds	r5, r5, #4
	cmp	r0, r1
	str	r0, [r6, #96]                   @ 4-byte Spill
	blo	.LBB0_39
.LBB0_37:                               @ %if.end216
                                        @ =>This Inner Loop Header: Depth=1
	ldr	r0, [r6, #68]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #48]                   @ 4-byte Reload
	ldr	r1, [r1, r5]
	bl	__aeabi_fsub
	mov	r4, r0
	str	r0, [r6, #92]                   @ 4-byte Spill
	ldr	r0, [r6, #64]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #44]                   @ 4-byte Reload
	ldr	r1, [r1, r5]
	bl	__aeabi_fsub
	mov	r1, r0
	str	r0, [r6, #104]                  @ 4-byte Spill
	mov	r0, r4
	bl	__aeabi_fmul
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r0, [r6, #72]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #52]                   @ 4-byte Reload
	ldr	r1, [r1, r5]
	bl	__aeabi_fsub
	mov	r4, r0
	str	r0, [r6, #100]                  @ 4-byte Spill
	ldr	r0, [r6, #60]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #40]                   @ 4-byte Reload
	ldr	r1, [r1, r5]
	bl	__aeabi_fsub
	mov	r1, r0
	mov	r0, r4
	mov	r4, r1
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fsub
	str	r0, [r6, #116]                  @ 4-byte Spill
	ldr	r0, [r6, #76]                   @ 4-byte Reload
	ldr	r0, [r0, r5]
	ldr	r1, [r6, #56]                   @ 4-byte Reload
	ldr	r1, [r1, r5]
	bl	__aeabi_fsub
	str	r0, [r6, #88]                   @ 4-byte Spill
	mov	r1, r4
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	ldr	r1, [r6, #104]                  @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	str	r1, [r6, #104]                  @ 4-byte Spill
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	str	r0, [r6, #84]                   @ 4-byte Spill
	ldr	r0, [r6, #88]                   @ 4-byte Reload
	ldr	r1, [r6, #92]                   @ 4-byte Reload
	bl	__aeabi_fmul
	mov	r4, r0
	ldr	r0, [r6, #100]                  @ 4-byte Reload
	mov	r1, r0
	bl	__aeabi_fmul
	mov	r1, r0
	mov	r0, r4
	bl	__aeabi_fsub
	mov	r1, r0
	ldr	r0, [r6, #84]                   @ 4-byte Reload
	str	r1, [r6, #100]                  @ 4-byte Spill
	bl	__aeabi_fdiv
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	mov	r4, r0
	ldr	r0, [r6, #80]                   @ 4-byte Reload
	ldr	r1, [r0, r5]
	mov	r0, r4
	bl	__aeabi_fcmplt
	cmp	r0, #0
	beq	.LBB0_36
@ %bb.38:                               @ %if.then248
                                        @   in Loop: Header=BB0_37 Depth=1
	ldr	r0, [r6, #80]                   @ 4-byte Reload
	str	r4, [r0, r5]
	b	.LBB0_36
.LBB0_39:                               @ %for.cond209.for.cond256.preheader_crit_edge
	ldr	r3, [r6, #32]                   @ 4-byte Reload
	ldr	r4, [r6, #4]                    @ 4-byte Reload
	subs	r0, r3, r4
	ldr	r2, [r6, #120]                  @ 4-byte Reload
	ldr	r5, [r6, #28]                   @ 4-byte Reload
	cmp	r0, r3
	bge	.LBB0_46
.LBB0_40:                               @ %for.body259.lr.ph
	cmp	r3, r5
	mov	r0, r3
	bgt	.LBB0_42
@ %bb.41:                               @ %for.body259.lr.ph
	mov	r0, r5
.LBB0_42:                               @ %for.body259.lr.ph
	adds	r1, r4, r3
	subs	r3, r1, r0
	lsls	r0, r0, #2
	lsls	r1, r4, #2
	subs	r0, r0, r1
	ldr	r1, [r6, #24]                   @ 4-byte Reload
	adds	r5, r1, r0
	b	.LBB0_44
.LBB0_43:                               @ %for.inc271
                                        @   in Loop: Header=BB0_44 Depth=1
	ldr	r0, [r6, #108]                  @ 4-byte Reload
	lsls	r1, r0, #23
	ldr	r0, [r6, #120]                  @ 4-byte Reload
	bl	__aeabi_fadd
	mov	r2, r0
	adds	r5, r5, #4
	ldr	r0, [r6, #96]                   @ 4-byte Reload
	subs	r3, r0, #1
	beq	.LBB0_46
.LBB0_44:                               @ %for.body259
                                        @ =>This Inner Loop Header: Depth=1
	str	r3, [r6, #96]                   @ 4-byte Spill
	ldr	r0, [r6, #104]                  @ 4-byte Reload
	str	r2, [r6, #120]                  @ 4-byte Spill
	mov	r1, r2
	bl	__aeabi_fmul
	mov	r1, r0
	ldr	r0, [r6, #116]                  @ 4-byte Reload
	bl	__aeabi_fadd
	ldr	r1, [r6, #100]                  @ 4-byte Reload
	bl	__aeabi_fdiv
	ldr	r1, [r6, #112]                  @ 4-byte Reload
	bl	__aeabi_fsub
	mov	r4, r0
	ldr	r1, [r5]
	bl	__aeabi_fcmplt
	cmp	r0, #0
	beq	.LBB0_43
@ %bb.45:                               @ %if.then267
                                        @   in Loop: Header=BB0_44 Depth=1
	str	r4, [r5]
	b	.LBB0_43
.LBB0_46:                               @ %for.end274
	subs	r6, r7, #7
	subs	r6, #5
	mov	sp, r6
	pop	{r4, r5, r6, r7, pc}
.Lfunc_end0:
	.size	bark_noise_hybridmp, .Lfunc_end0-bark_noise_hybridmp
	.cantunwind
	.fnend
                                        @ -- End function
	.ident	"clang version 3.6.0 "
	.section	".note.GNU-stack","",%progbits
	.eabi_attribute	30, 2	@ Tag_ABI_optimization_goals
