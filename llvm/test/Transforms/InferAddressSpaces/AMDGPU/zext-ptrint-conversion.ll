; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -o - -passes=infer-address-spaces %s | FileCheck -check-prefixes=COMMON,AMDGCN %s
; RUN: opt -S -o - -passes=infer-address-spaces -assume-default-is-flat-addrspace %s | FileCheck -check-prefixes=COMMON,NOTTI %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7:8"

; COMMON-LABEL: @zext_ptrint_conversion(
; AMDGCN-NEXT: store i32 0, ptr addrspace(3) %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i64
; NOTTI-NEXT: zext i32 %{{.*}} to i64
; NOTTI-NEXT: inttoptr i64 %{{.*}} to ptr
; NOTTI-NEXT: store i32 0, ptr %{{.*}}
; NOTTI-NEXT: ret void
define void @zext_ptrint_conversion(ptr addrspace(3) %x) {
  %1 = ptrtoint ptr addrspace(3) %x to i32
  %2 = zext i32 %1 to i64
  %3 = inttoptr i64 %2 to ptr
  store i32 0, ptr %3
  ret void
}

; COMMON-LABEL: @non_zext_ptrint_conversion(
; AMDGCN-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i16
; AMDGCN-NEXT: zext i16 %{{.*}} to i64
; AMDGCN-NEXT: inttoptr i64 %{{.*}} to ptr
; AMDGCN-NEXT: store i32 0, ptr %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i16
; NOTTI-NEXT: zext i16 %{{.*}} to i64
; NOTTI-NEXT: inttoptr i64 %{{.*}} to ptr
; NOTTI-NEXT: store i32 0, ptr %{{.*}}
; NOTTI-NEXT: ret void
define void @non_zext_ptrint_conversion(ptr addrspace(3) %x) {
  %1 = ptrtoint ptr addrspace(3) %x to i16
  %2 = zext i16 %1 to i64
  %3 = inttoptr i64 %2 to ptr
  store i32 0, ptr %3
  ret void
}

; COMMON-LABEL: @non_zext_ptrint_conversion2(
; AMDGCN-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i32
; AMDGCN-NEXT: zext i32 %{{.*}} to i128
; AMDGCN-NEXT: inttoptr i128 %{{.*}} to ptr
; AMDGCN-NEXT: store i32 0, ptr %{{.*}}
; AMDGCN-NEXT: ret void
; NOTTI-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i32
; NOTTI-NEXT: zext i32 %{{.*}} to i128
; NOTTI-NEXT: inttoptr i128 %{{.*}} to ptr
; NOTTI-NEXT: store i32 0, ptr %{{.*}}
; NOTTI-NEXT: ret void
define void @non_zext_ptrint_conversion2(ptr addrspace(3) %x) {
  %1 = ptrtoint ptr addrspace(3) %x to i32
  %2 = zext i32 %1 to i128
  %3 = inttoptr i128 %2 to ptr
  store i32 0, ptr %3
  ret void
}
