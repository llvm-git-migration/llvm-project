// REQUIRES: x86
// UNSUPPORTED: system-windows

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(.text .text1); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: not ld.lld %t.o -o %t --script %t.script 2>&1 | FileCheck -check-prefix=ERR %s
// ERR: ld.lld: error: {{.*}}.o:(.text+{{.*}}): prohibited cross reference from .text to in .text1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS_TO(.text1 .text); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: not ld.lld %t.o -o %t --script %t.script 2>&1 | FileCheck -check-prefix=ERR1 %s
// ERR1: ld.lld: error: {{.*}}.o:(.text+{{.*}}): prohibited cross reference from .text to in .text1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(.text1 .text .text2); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: not ld.lld %t.o -o %t --script %t.script 2>&1 | FileCheck -check-prefix=ERR2 %s
// ERR2: ld.lld: error: {{.*}}.o:(.text+{{.*}}): prohibited cross reference from .text to in .text1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS_TO(.text .text1); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS_TO(); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(.text); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS_TO(.text); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS_TO(.text2 .text); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(.text .text2); \
// RUN:       SECTIONS { \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

// RUN: llvm-mc -filetype=obj -o %t.o %s
// RUN: echo "NOCROSSREFS(.text .text2); \
// RUN:       SECTIONS { \
// RUN:          foo = ABSOLUTE(.); \
// RUN:         .text  : { *(.text) } \
// RUN:         .text1 : { *(.text1) } \
// RUN:         .text2 : { *(.text2) } \
// RUN: } " > %t.script
// RUN: ld.lld %t.o -o %t --script %t.script 2>&1

.global _start
_start:
	call test

	.type	unused,@object
	.comm	unused,4,4

	.section	.noalloc,"",@progbits
	.quad	unused

.section .text
test:
	call test1

.section .text2
test2:
	.reloc ., R_X86_64_32, foo
	nop

.section .text1
test1:
	nop
