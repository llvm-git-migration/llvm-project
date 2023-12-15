// REQUIRES: systemz
// RUN: llvm-mc %s -o %t.o -triple s390x-unknown-linux -filetype=obj
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s
// RUN: ld.lld --noinhibit-exec -shared %t.o -o %t 2>&1 | FileCheck %s
// RUN: ls %t

// CHECK: {{.*}}:(.text+0x2): relocation R_390_PC16DBL out of range: 65536 is not in [-65536, 65535]
// CHECK-NOT: relocation

        j    foo
        j    foo

        .hidden foo
        .bss
        .zero 0xdf88
foo:
