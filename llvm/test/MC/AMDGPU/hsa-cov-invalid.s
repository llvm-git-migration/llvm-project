// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple amdgcn-amd-mesa3d -mcpu=gfx802 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple amdgcn-amd- -mcpu=gfx802 %s 2>&1 | FileCheck %s

// CHECK: error: unknown directive
.amdhsa_code_object_version 4
