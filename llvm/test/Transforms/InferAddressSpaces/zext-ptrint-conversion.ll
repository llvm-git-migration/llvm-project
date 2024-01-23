; RUN: opt -S -o - -passes=infer-address-spaces -assume-default-is-flat-addrspace %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7:8"

; CHECK-LABEL: @zext_ptrint_conversion(
; CHECK-NEXT: store i32 0, ptr addrspace(3) %{{.*}}
; CHECK-NEXT: ret void
define void @zext_ptrint_conversion(ptr addrspace(3) %x) {
  %tmp1 = ptrtoint ptr addrspace(3) %x to i32
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = inttoptr i64 %tmp2 to ptr
  store i32 0, ptr %tmp3
  ret void
}

; CHECK-LABEL: @non_zext_ptrint_conversion(
; CHECK-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i16
; CHECK-NEXT: zext i16 %{{.*}} to i64
; CHECK-NEXT: inttoptr i64 %{{.*}} to ptr
; CHECK-NEXT: store i32 0, ptr %{{.*}}
; CHECK-NEXT: ret void
define void @non_zext_ptrint_conversion(ptr addrspace(3) %x) {
  %tmp1 = ptrtoint ptr addrspace(3) %x to i16
  %tmp2 = zext i16 %tmp1 to i64
  %tmp3 = inttoptr i64 %tmp2 to ptr
  store i32 0, ptr %tmp3
  ret void
}

; CHECK-LABEL: @non_zext_ptrint_conversion2(
; CHECK-NEXT: ptrtoint ptr addrspace(3) %{{.*}} to i32
; CHECK-NEXT: zext i32 %{{.*}} to i128
; CHECK-NEXT: inttoptr i128 %{{.*}} to ptr
; CHECK-NEXT: store i32 0, ptr %{{.*}}
; CHECK-NEXT: ret void
define void @non_zext_ptrint_conversion2(ptr addrspace(3) %x) {
  %tmp1 = ptrtoint ptr addrspace(3) %x to i32
  %tmp2 = zext i32 %tmp1 to i128
  %tmp3 = inttoptr i128 %tmp2 to ptr
  store i32 0, ptr %tmp3
  ret void
}
