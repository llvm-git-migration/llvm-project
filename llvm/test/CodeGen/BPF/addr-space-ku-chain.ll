; RUN: opt --bpf-check-and-opt-ir -S -mtriple=bpf-pc-linux < %s | FileCheck %s

; Generated from the following C code:
;
;   #define __uptr __attribute__((address_space(272)))
;
;   void test(void __uptr *q, void __uptr *p) {
;     void __uptr * __uptr *a;
;     void __uptr * __uptr *b;
;
;      a = q + 8;
;     *a = p;
;      b = p + 16;
;     *b = a;
;   }
;
; Using the following command:
;
;   clang --target=bpf -O2 -S -emit-llvm -o t.ll t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @test(ptr addrspace(1) noundef %q, ptr addrspace(1) noundef %p) local_unnamed_addr #0 {
; CHECK-LABEL: define dso_local void @test
; CHECK-SAME:    (ptr addrspace(1) noundef [[Q:%.*]], ptr addrspace(1) noundef [[P:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[QU:%.*]] = call ptr addrspace(1) @llvm.bpf.arena.cast.p1.p1(ptr addrspace(1) [[Q]], i32 2)
; CHECK-NEXT:    [[PK:%.*]] = call ptr addrspace(1) @llvm.bpf.arena.cast.p1.p1(ptr addrspace(1) [[P]], i32 1)
; CHECK-NEXT:    [[PU:%.*]] = call ptr addrspace(1) @llvm.bpf.arena.cast.p1.p1(ptr addrspace(1) [[P]], i32 2)
; CHECK-NEXT:    [[QK:%.*]] = call ptr addrspace(1) @llvm.bpf.arena.cast.p1.p1(ptr addrspace(1) [[Q]], i32 1)
; CHECK-NEXT:    [[AU:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[QU]], i64 8
; CHECK-NEXT:    [[AK:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[QK]], i64 8
; CHECK-NEXT:    store ptr addrspace(1) [[PU]], ptr addrspace(1) [[AK]], align 8
; CHECK-NEXT:    [[BK:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[PK]], i64 16
; CHECK-NEXT:    store ptr addrspace(1) [[AU]], ptr addrspace(1) [[BK]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %add.ptr = getelementptr inbounds i8, ptr addrspace(1) %q, i64 8
  store ptr addrspace(1) %p, ptr addrspace(1) %add.ptr, align 8, !tbaa !3
  %add.ptr1 = getelementptr inbounds i8, ptr addrspace(1) %p, i64 16
  store ptr addrspace(1) %add.ptr, ptr addrspace(1) %add.ptr1, align 8, !tbaa !3
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"some clan version"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
