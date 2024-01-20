// RUN: llvm-mc --triple=amdgcn --mcpu=gfx904 %s | FileCheck %s
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx940 %s | FileCheck %s
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1010 %s | FileCheck %s
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1030 %s | FileCheck %s
// RUN: llvm-mc --triple=amdgcn --mcpu=gfx1100 %s | FileCheck %s

.text
  v_writelane_b32 v1, s13, m0

// CHECK: v_writelane_b32 v1, s13, m0
