# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t

## Check for incompatible cases
# RUN: not ld.lld %t --irpgo-profile=/dev/null --bp-startup-sort=function --call-graph-ordering-file=/dev/null 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-CALLGRAPH-ERR
# RUN: not ld.lld --bp-compression-sort=function --call-graph-ordering-file /dev/null 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-CALLGRAPH-ERR
# RUN: not ld.lld --bp-startup-sort=function 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-ERR
# RUN: not ld.lld --bp-compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=BP-STARTUP-COMPRESSION-ERR
# RUN: not ld.lld --bp-startup-sort=invalid --bp-compression-sort=invalid 2>&1 | FileCheck %s --check-prefix=BP-INVALID

# BP-STARTUP-CALLGRAPH-ERR: error: --bp-startup-sort=function is incompatible with --call-graph-ordering-file
# BP-COMPRESSION-CALLGRAPH-ERR: error: --bp-compression-sort is incompatible with --call-graph-ordering-file
# BP-STARTUP-ERR: error: --bp-startup-sort=function must be used with --irpgo-profile
# BP-STARTUP-COMPRESSION-ERR: error: --bp-compression-sort-startup-functions must be used with --irpgo-profile

# BP-INVALID: error: --bp-compression-sort=: expected [none|function|data|both]
# BP-INVALID: error: --bp-startup-sort=: expected [none|function]

# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-profdata merge a.proftext -o a.profdata
# RUN: ld.lld a.o --irpgo-profile=a.profdata --bp-startup-sort=function --verbose-bp-section-orderer --icf=all 2>&1 | FileCheck %s --check-prefix=STARTUP-FUNC-ORDER

# STARTUP-FUNC-ORDER: Ordered 3 sections using balanced partitioning
# STARTUP-FUNC-ORDER: Total area under the page fault curve: 3.

# RUN: ld.lld a.o --irpgo-profile=a.profdata --bp-startup-sort=function
# RUN: llvm-nm -jn a.out | tr '\n' , | FileCheck %s --check-prefix=STARTUP
# STARTUP: s1,s2,s3,A,B,C,F,E,D,_start,r1,r2,r3,r4,

# RUN: ld.lld a.o --irpgo-profile=a.profdata --bp-startup-sort=function --symbol-ordering-file a.txt
# RUN: llvm-nm -jn a.out | tr '\n' , | FileCheck %s --check-prefix=ORDER-STARTUP
# ORDER-STARTUP: s2,s1,s3,A,F,E,D,B,C,_start,r3,r2,r1,r4,

## Rodata
# ORDERFILE:      s2
# ORDERFILE-NEXT: s1
# ORDERFILE-NEXT: s3

## Functions
# ORDERFILE-NEXT: A
# ORDERFILE-NEXT: F
# ORDERFILE-NEXT: E
# ORDERFILE-NEXT: D
# ORDERFILE-DAG:  B
# ORDERFILE-DAG:  C
# ORDERFILE-NEXT: _start

## Data
# ORDERFILE-NEXT: r3
# ORDERFILE-NEXT: r2
# ORDERFILE-NEXT: r1
# ORDERFILE-NEXT: r4

# RUN: ld.lld a.o --verbose-bp-section-orderer --bp-compression-sort=function 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-FUNC
# RUN: ld.lld a.o --verbose-bp-section-orderer --bp-compression-sort=data 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-DATA
# RUN: ld.lld a.o --verbose-bp-section-orderer --bp-compression-sort=both 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-BOTH
# RUN: ld.lld a.o --verbose-bp-section-orderer --bp-compression-sort=both --irpgo-profile=a.profdata --bp-startup-sort=function 2>&1 | FileCheck %s --check-prefix=BP-COMPRESSION-BOTH

# BP-COMPRESSION-FUNC: Ordered 7 sections using balanced partitioning
# BP-COMPRESSION-DATA: Ordered 7 sections using balanced partitioning
# BP-COMPRESSION-BOTH: Ordered 14 sections using balanced partitioning

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
A, B, C

A
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

B
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

C
# Func Hash:
3333
# Num Counters:
1
# Counter Values:
1

D
# Func Hash:
4444
# Num Counters:
1
# Counter Values:
1

#--- a.txt
A
F
E
D
s2
s1
r3
r2

#--- a.c
const char s1[] = "hello world";
const char s2[] = "i am a string";
const char s3[] = "this is s3";
const char *r1 = s1;
const char **r2 = &r1;
const char ***r3 = &r2;
const char *r4 = s2;

int C(int a);
int B(int a);
void A();

int F(int a) { return C(a + 3); }
int E(int a) { return C(a + 2); }
int D(int a) { return B(a + 2); }
int C(int a) { A(); return a + 2; }
int B(int a) { A(); return a + 1; }
void A() {}

int _start() { return 0; }

#--- gen
clang --target=aarch64-linux-gnu -O0 -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -S a.c -o -
;--- a.s
	.file	"a.c"
	.section	.text.F,"ax",@progbits
	.globl	F                               // -- Begin function F
	.p2align	2
	.type	F,@function
F:                                      // @F
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #3
	bl	C
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end0:
	.size	F, .Lfunc_end0-F
                                        // -- End function
	.section	.text.C,"ax",@progbits
	.globl	C                               // -- Begin function C
	.p2align	2
	.type	C,@function
C:                                      // @C
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	bl	A
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end1:
	.size	C, .Lfunc_end1-C
                                        // -- End function
	.section	.text.E,"ax",@progbits
	.globl	E                               // -- Begin function E
	.p2align	2
	.type	E,@function
E:                                      // @E
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	C
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end2:
	.size	E, .Lfunc_end2-E
                                        // -- End function
	.section	.text.D,"ax",@progbits
	.globl	D                               // -- Begin function D
	.p2align	2
	.type	D,@function
D:                                      // @D
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	ldur	w8, [x29, #-4]
	add	w0, w8, #2
	bl	B
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end3:
	.size	D, .Lfunc_end3-D
                                        // -- End function
	.section	.text.B,"ax",@progbits
	.globl	B                               // -- Begin function B
	.p2align	2
	.type	B,@function
B:                                      // @B
// %bb.0:                               // %entry
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	stur	w0, [x29, #-4]
	bl	A
	ldur	w8, [x29, #-4]
	add	w0, w8, #1
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end4:
	.size	B, .Lfunc_end4-B
                                        // -- End function
	.section	.text.A,"ax",@progbits
	.globl	A                               // -- Begin function A
	.p2align	2
	.type	A,@function
A:                                      // @A
// %bb.0:                               // %entry
	ret
.Lfunc_end5:
	.size	A, .Lfunc_end5-A
                                        // -- End function
	.section	.text._start,"ax",@progbits
	.globl	_start                          // -- Begin function _start
	.p2align	2
	.type	_start,@function
_start:                                 // @_start
// %bb.0:                               // %entry
	mov	w0, wzr
	ret
.Lfunc_end6:
	.size	_start, .Lfunc_end6-_start
                                        // -- End function
	.type	s1,@object                      // @s1
	.section	.rodata.s1,"a",@progbits
	.globl	s1
s1:
	.asciz	"hello world"
	.size	s1, 12

	.type	s2,@object                      // @s2
	.section	.rodata.s2,"a",@progbits
	.globl	s2
s2:
	.asciz	"i am a string"
	.size	s2, 14

	.type	s3,@object                      // @s3
	.section	.rodata.s3,"a",@progbits
	.globl	s3
s3:
	.asciz	"this is s3"
	.size	s3, 11

	.type	r1,@object                      // @r1
	.section	.data.r1,"aw",@progbits
	.globl	r1
	.p2align	3, 0x0
r1:
	.xword	s1
	.size	r1, 8

	.type	r2,@object                      // @r2
	.section	.data.r2,"aw",@progbits
	.globl	r2
	.p2align	3, 0x0
r2:
	.xword	r1
	.size	r2, 8

	.type	r3,@object                      // @r3
	.section	.data.r3,"aw",@progbits
	.globl	r3
	.p2align	3, 0x0
r3:
	.xword	r2
	.size	r3, 8

	.type	r4,@object                      // @r4
	.section	.data.r4,"aw",@progbits
	.globl	r4
	.p2align	3, 0x0
r4:
	.xword	s2
	.size	r4, 8

	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym C
	.addrsig_sym B
	.addrsig_sym A
	.addrsig_sym s1
	.addrsig_sym s2
	.addrsig_sym r1
	.addrsig_sym r2
