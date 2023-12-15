// REQUIRES: systemz
// RUN: llvm-mc %s -o %t.o -triple s390x-unknown-linux -mcpu=z13 -filetype=obj
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s
// RUN: ld.lld --noinhibit-exec -shared %t.o -o %t 2>&1 | FileCheck %s
// RUN: ls %t

// CHECK: {{.*}}:(.text+0x3): relocation R_390_PC24DBL out of range: 16777216 is not in [-16777216, 16777215]
// CHECK-NOT: relocation

        bprp 1,0,foo
        bprp 1,0,foo

        .hidden foo
	.section .text.foo
	.zero 0xfffff4
foo:
