// REQUIRES: systemz
// RUN: llvm-mc %s -o %t.o -triple s390x-unknown-linux -mcpu=z13 -filetype=obj
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s
// RUN: ld.lld --noinhibit-exec -shared %t.o -o %t 2>&1 | FileCheck %s
// RUN: ls %t

// CHECK: {{.*}}:(.text+0x1): relocation R_390_PC12DBL out of range: 4096 is not in [-4096, 4095]
// CHECK-NOT: relocation

        bprp 1,foo,0
        bprp 1,foo,0

        .hidden foo
	.section .text.foo
	.zero 0xff4
foo:
