; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -filetype=obj < %s > %t
; RUN: llvm-objdump -s -j .rodata %t | FileCheck --check-prefix=OBJDUMP %s

; OBJDUMP: Contents of section .rodata:
; OBJDUMP-NEXT: 0000 00000000 00000000 10010000 00000000
; OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0030 4000af00 94130000 1a000400 00000000
; OBJDUMP-NOT: 0030 0000af00 94130000 1a000400 00000000

; Include preloaded SGPRs that are not explicitly used in the kernel in
; GRANULATED_WAVEFRONT_SGPR_COUNT.

define amdgpu_kernel void @amdhsa_kernarg_preload_num_sgprs(i128 inreg) { ret void }
