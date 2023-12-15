// REQUIRES: systemz
// RUN: llvm-mc %s -o %t.o -triple s390x-unknown-linux -filetype=obj
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s
// RUN: ld.lld --noinhibit-exec -shared %t.o -o %t 2>&1 | FileCheck %s
// RUN: ls %t

// CHECK: {{.*}}:(.text+0x2): relocation R_390_PC32DBL out of range: 4294967296 is not in [-4294967296, 4294967295]
// CHECK-NOT: relocation

        larl    %r0, foo
        larl    %r0, foo

        .hidden foo
        .bss
        .zero 0xffffdf80
foo:
