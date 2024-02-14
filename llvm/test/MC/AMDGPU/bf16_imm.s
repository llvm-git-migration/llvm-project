// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck %s

v_dot2_bf16_bf16 v5, v1, v2, 100.0
// CHECK: v_dot2_bf16_bf16 v5, v1, v2, 0x42c8 ; encoding: [0x05,0x00,0x67,0xd6,0x01,0x05,0xfe,0x03,0xc8,0x42,0x00,0x00]

v_dot2_bf16_bf16 v5, v1, v2, 1.0
// CHECK: v_dot2_bf16_bf16 v5, v1, v2, 1.0 ; encoding: [0x05,0x00,0x67,0xd6,0x01,0x05,0xca,0x03]
