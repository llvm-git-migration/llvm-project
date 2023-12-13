; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module %s -o %t.ll
; RUN: not --crash opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module %t.ll -o - 2>&1 | FileCheck %s --check-prefix=ERR

; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module --amdgpu-lower-module-lds-force-add-moduleflag=1 %s -o %t.ll
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module %t.ll -o - | FileCheck %s

; Check re-run of LowerModuleLDS don't crash when the module flag is used.
;
; We first check this test still crashes when ran twice. If it no longer crashes at some point
; we should update it to ensure the flag still does its job.
;
; This test jus has the bare minimum checks to see if the pass ran.

; ERR: LLVM ERROR: LDS variables with absolute addresses are unimplemented.

; CHECK: %llvm.amdgcn.module.lds.t = type { float, [4 x i8], i32 }
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t poison, align 8

; CHECK: attributes #0 = { "amdgpu-lds-size"="12" }

@var0 = addrspace(3) global float poison, align 8
@var1 = addrspace(3) global i32 poison, align 8
@ptr = addrspace(1) global ptr addrspace(3) @var1, align 4
@with_init = addrspace(3) global i64 0

define void @func() {
  %dec = atomicrmw fsub ptr addrspace(3) @var0, float 1.0 monotonic
  %val0 = load i32, ptr addrspace(3) @var1, align 4
  %val1 = add i32 %val0, 4
  store i32 %val1, ptr addrspace(3) @var1, align 4
  %unused0 = atomicrmw add ptr addrspace(3) @with_init, i64 1 monotonic
  ret void
}

define amdgpu_kernel void @kern_call() {
  call void @func()
  %dec = atomicrmw fsub ptr addrspace(3) @var0, float 2.0 monotonic
  ret void
}
