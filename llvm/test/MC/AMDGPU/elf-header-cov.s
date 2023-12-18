// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS4

// RUN: sed 's/COV/5/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS5

// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=PAL

// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd-mesa3d -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=MSA

// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd- -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=UNK

.amdgcn_code_object_version COV

// HS4: OS/ABI: AMDGPU_HSA (0x40)
// HS4-NEXT: ABIVersion: 2

// HS5: OS/ABI: AMDGPU_HSA (0x40)
// HS5-NEXT: ABIVersion: 3

// PAL: OS/ABI: AMDGPU_PAL (0x41)
// PAL-NEXT: ABIVersion: 0

// MSA: OS/ABI: AMDGPU_MESA3D (0x42)
// MSA-NEXT: ABIVersion: 0

// UNK: OS/ABI: SystemV (0x0)
// UNK-NEXT: ABIVersion: 0
