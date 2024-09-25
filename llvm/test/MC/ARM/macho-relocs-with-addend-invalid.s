// RUN: not llvm-mc -triple armv7-apple-darwin -filetype=obj %s 2>&1 | FileCheck %s

// Check that the relocation size is valid.
// Check outside of range of the largest accepted positive number
_foo1:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo1+33554432

// Check Same as above, for smallest negative value
_foo2:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo2-33554436

// Edge case - subtracting positive number
_foo3:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo3-0x2000010

// Edge case - adding negative number
_foo4:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo4+0x2000008

_foo5:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    bl  _foo5+33554432

_foo6:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    blx _foo6+33554432

// blx instruction is aligned to 16-bits.
_foo7:
    // CHECK-NOT:[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    blx _foo6+33554430

_foo8:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    ble _foo8+33554432

_foo9:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    beq _foo9+33554432

    // Check that the relocation alignment is valid.
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    bl  _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    blx _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    b   _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    ble _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    beq _foo1+0x101
