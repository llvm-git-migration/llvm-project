; RUN: llc -mtriple=thumbv7em-eabi -verify-machineinstrs %s -o - | FileCheck %s

@d = external global [64 x i32]
@s = external global [64 x i32]

; Function Attrs: nounwind
define void @t1() #0 {
entry:
; CHECK-LABEL: t1:
; We use '[rl0-9]+' to allow 'r0'..'r12', 'lr'
; CHECK: movt [[LB:[rl0-9]+]], :upper16:d
; CHECK: movt [[SB:[rl0-9]+]], :upper16:s
; CHECK-NOT: ldm
; CHECK-NOT: stm
    tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([64 x i32]* @s to i8*), i8* bitcast ([64 x i32]* @d to i8*), i32 17, i32 4, i1 false)
    ret void
}

; Function Attrs: nounwind
define void @t2() #0 {
entry:
; CHECK-LABEL: t2:
; CHECK: movt [[LB:[rl0-9]+]], :upper16:d
; CHECK: movt [[SB:[rl0-9]+]], :upper16:s
; CHECK-NOT: ldm
; CHECK-NOT: stm
    tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([64 x i32]* @s to i8*), i8* bitcast ([64 x i32]* @d to i8*), i32 15, i32 4, i1 false)
    ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #1
