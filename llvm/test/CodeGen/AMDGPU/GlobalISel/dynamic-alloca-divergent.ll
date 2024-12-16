; RUN: not llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel-abort=2 -pass-remarks-missed="gisel.*" -verify-machineinstrs -o /dev/null 2>&1 %s | FileCheck -check-prefix=ERR %s

; ERR: remark: <unknown>:0:0: cannot select: %{{[0-9]+}}:sreg_32(p5) = G_DYN_STACKALLOC %{{[0-9]+}}:vgpr(s32), 1 (in function: kernel_dynamic_stackalloc_vgpr_align4)
; ERR-NEXT: warning: Instruction selection used fallback path for kernel_dynamic_stackalloc_vgpr_align4
; ERR-NEXT: error: <unknown>:0:0: in function kernel_dynamic_stackalloc_vgpr_align4 void (ptr addrspace(1)): unsupported dynamic alloca

; ERR: remark: <unknown>:0:0: cannot select: %{{[0-9]+}}:sreg_32(p5) = G_DYN_STACKALLOC %{{[0-9]+}}:vgpr(s32), 1 (in function: func_dynamic_stackalloc_vgpr_align4)
; ERR-NEXT: warning: Instruction selection used fallback path for func_dynamic_stackalloc_vgpr_align4
; ERR-NEXT: error: <unknown>:0:0: in function func_dynamic_stackalloc_vgpr_align4 void (i32): unsupported dynamic alloca

define amdgpu_kernel void @kernel_dynamic_stackalloc_vgpr_align4(ptr addrspace(1) %ptr) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, ptr addrspace(1) %ptr, i32 %id
  %n = load i32, ptr addrspace(1) %gep
  %alloca = alloca i32, i32 %n, align 4, addrspace(5)
  store volatile ptr addrspace(5) %alloca, ptr addrspace(1) undef
  ret void
}

define void @func_dynamic_stackalloc_vgpr_align4(i32 %n) {
  %alloca = alloca i32, i32 %n, align 4, addrspace(5)
  store volatile ptr addrspace(5) %alloca, ptr addrspace(1) undef
  ret void
}

define void @func_dynamic_stackalloc_vgpr_align32(i32 %n) {
  %alloca = alloca i32, i32 %n, align 32, addrspace(5)
  store volatile ptr addrspace(5) %alloca, ptr addrspace(1) undef
  ret void
}

define amdgpu_kernel void @kernel_non_entry_block_dynamic_alloca(ptr addrspace(1) %out, i32 %arg.cond, i32 %in) {
    entry:
    %cond = icmp eq i32 %arg.cond, 0
    br i1 %cond, label %bb.0, label %bb.1

    bb.0:
    %alloca = alloca i32, i32 %in, align 64, addrspace(5)
    %gep1 = getelementptr i32, ptr addrspace(5) %alloca, i32 1
    store volatile i32 0, ptr addrspace(5) %alloca
    store volatile i32 1, ptr addrspace(5) %gep1
    br label %bb.1

    bb.1:
    ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
