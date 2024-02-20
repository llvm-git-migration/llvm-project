; RUN: opt < %s -x86-flags-copy-lowering -S | FileCheck %s
; Test that the 'cld' instruction is created only when needed.

define void @test1() {
; CHECK-LABEL: @test1
; CHECK-NOT: call void asm sideeffect "cld"
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2
; CHECK: call void asm sideeffect "cld"
  call void @external()
  ret void
}

declare void @external()

define void @test3() {
; CHECK-LABEL: @test3
; CHECK: call void asm sideeffect "cld"
  %1 = alloca [100 x i8], align 16
  %2 = getelementptr inbounds [100 x i8], [100 x i8]* %1, i64 0, i64 0
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 dereferenceable(100) %2, i8 0, i64 100, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1
