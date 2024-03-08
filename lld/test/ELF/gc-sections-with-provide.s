# REQUIRES: x86

# This test verifies that garbage-collection is correctly garbage collecting
# unused sections when the symbol of the unused section is only referred by
# an unused PROVIDE symbol.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: ld.lld -o a_nogc a.o -T script.t
# RUN: llvm-readobj --sections --symbols a_nogc | FileCheck -check-prefix=NOGC %s
# RUN: ld.lld -o a_gc a.o --gc-sections --print-gc-sections -T script.t | FileCheck --check-prefix=GC_LINK %s
# RUN: llvm-readobj --sections --symbols a_gc | FileCheck -check-prefix=GC %s

NOGC: Name: foo
NOGC: Name: used
NOGC: Name: bar
NOGC: Name: baz
NOGC-NOT: unused

GC_LINK: removing unused section a.o:(.text.bar)

GC: Name: foo
GC: Name: used
GC: Name: baz
GC-NOT: bar
GC-NOT: unused

#--- a.s
.global _start
_start:
 call foo
 call used

.section .text.foo,"ax",@progbits
foo:
 nop

.section .text.bar,"ax",@progbits
.global bar
bar:
 nop

.section .text.baz,"ax",@progbits
.global baz
baz:
 nop


#--- script.t
PROVIDE(unused = bar);
PROVIDE(used = baz);