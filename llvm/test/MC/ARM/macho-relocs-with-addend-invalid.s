// RUN: llvm-mc -triple armv7-apple-darwin -filetype=obj %s | llvm-objdump -d --macho - | FileCheck %s

_foo:
    // First issue: relocation addend range not checked, silently truncated
    // CHECK: bl      0xffffff00
    // CHECK: blx     0xffffff00
    // CHECK: b       0xffffff00
    // CHECK: ble     0xffffff00
    // CHECK: beq     0xffffff00

    bl  _foo+0xfffffff00
    blx _foo+0xfffffff00
    b   _foo+0xfffffff00
    ble _foo+0xfffffff00
    beq _foo+0xfffffff00

    // Second bug: relocation addend alignment not checked, silently rounded
    // CHECK: bl      0x100
    // CHECK: blx     0x100
    // CHECK: b       0x100
    // CHECK: ble     0x100
    // CHECK: beq     0x100
    
    bl  _foo+0x101
    blx _foo+0x101
    b   _foo+0x101
    ble _foo+0x101
    beq _foo+0x101
