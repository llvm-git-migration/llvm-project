// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS4

// RUN: sed 's/COV/5/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS5

.amdhsa_code_object_version COV

// HS4: OS/ABI: AMDGPU_HSA (0x40)
// HS4-NEXT: ABIVersion: 2

// HS5: OS/ABI: AMDGPU_HSA (0x40)
// HS5-NEXT: ABIVersion: 3
