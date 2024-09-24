// RUN: not llvm-mc -triple armv7-apple-darwin -filetype=obj %s 2>&1 | FileCheck %s

_foo:
    // Check that the relocation size is valid.

    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    bl  _foo+0xfffffff00
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    blx _foo+0xfffffff00
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo+0xfffffff00
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    ble _foo+0xfffffff00
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    beq _foo+0xfffffff00

    // Check that the relocation alignment is valid.

    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    bl  _foo+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    blx _foo+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    b   _foo+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    ble _foo+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    beq _foo+0x101
