; RUN: llc -filetype=asm < %s -mcpu=mips2 | FileCheck %s -check-prefixes=MIPS2
;
; Created from the following test case (PR121463) with
; clang -cc1 -triple mips-unknown-freebsd -target-cpu mips2 -O2 -emit-llvm test.c -o test.ll
; int l2arc_feed_secs, l2arc_feed_min_ms, l2arc_write_interval_wrote, l2arc_write_interval_next;
; void l2arc_write_interval() {
;   int interval;
;   if (l2arc_write_interval_wrote)
;     interval = l2arc_feed_min_ms / l2arc_feed_secs;
;   l2arc_write_interval_next = interval;
; }

target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-freebsd"

@l2arc_write_interval_wrote = local_unnamed_addr global i32 0, align 4
@l2arc_feed_min_ms = local_unnamed_addr global i32 0, align 4
@l2arc_feed_secs = local_unnamed_addr global i32 0, align 4
@l2arc_write_interval_next = local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local void @l2arc_write_interval() local_unnamed_addr #0 {
; MIPS2-LABEL: l2arc_write_interval:
; MIPS2:       # %bb.0: # %entry
; MIPS2-NEXT:    lui $1, %hi(l2arc_write_interval_wrote)
; MIPS2-NEXT:    lw $1, %lo(l2arc_write_interval_wrote)($1)
; MIPS2-NEXT:    beqz $1, $BB0_2
; MIPS2-NEXT:    nop
; MIPS2-NEXT:  # %bb.1: # %if.then
; MIPS2-NEXT:    lui $1, %hi(l2arc_feed_secs)
; MIPS2-NEXT:    lw $1, %lo(l2arc_feed_secs)($1)
; MIPS2-NEXT:    lui $2, %hi(l2arc_feed_min_ms)
; MIPS2-NEXT:    lw $2, %lo(l2arc_feed_min_ms)($2)
; MIPS2-NEXT:    div $zero, $2, $1
; MIPS2-NEXT:    teq $1, $zero, 7
; MIPS2-NEXT:    mflo $2
; MIPS2-NEXT:    j $BB0_3
; MIPS2-NEXT:    nop
entry:
  %0 = load i32, ptr @l2arc_write_interval_wrote, align 4, !tbaa !2
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %1 = load i32, ptr @l2arc_feed_min_ms, align 4, !tbaa !2
  %2 = load i32, ptr @l2arc_feed_secs, align 4, !tbaa !2
  %div = sdiv i32 %1, %2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %interval.0 = phi i32 [ %div, %if.then ], [ undef, %entry ]
  store i32 %interval.0, ptr @l2arc_write_interval_next, align 4, !tbaa !2
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="mips2" "target-features"="+mips2" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 20.0.0git (git@github.com:yingopq/llvm-project.git c23f2417dc5f6dc371afb07af5627ec2a9d373a0)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

