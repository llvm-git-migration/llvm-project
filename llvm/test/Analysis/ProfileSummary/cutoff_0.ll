; RUN: opt < %s -disable-output -passes=print-profile-summary -S 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output -profile-summary-cutoff-hot=0 -passes=print-profile-summary -S 2>&1 | FileCheck %s -check-prefixes=HOT-CUTOFF-0
; RUN: opt < %s -disable-output -profile-summary-cutoff-cold=0 -profile-summary-hot-count=18446744073709551615 -passes=print-profile-summary -S 2>&1 | FileCheck %s -check-prefixes=COLD-CUTOFF-0

define void @f1() !prof !20 {
; CHECK-LABEL: f1 :hot
; HOT-CUTOFF-0-LABEL: f1{{$}}
; COLD-CUTOFF-0-LABEL: f1 :cold

  ret void
}

define void @f2() !prof !21 {
; CHECK-LABEL: f2 :cold
; HOT-CUTOFF-0-LABEL: f2 :cold
; COLD-CUTOFF-0-LABEL: f2 :cold

  ret void
}

define void @f3() !prof !22 {
; CHECK-LABEL: f3 :hot
; HOT-CUTOFF-0-LABEL: f3{{$}}
; COLD-CUTOFF-0-LABEL: f3 :cold

  ret void
}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 400}
!21 = !{!"function_entry_count", i64 1}
!22 = !{!"function_entry_count", i64 100}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, i64 18446744073709551615, i32 0}
!13 = !{i32 10000, i64 100, i32 1}
!14 = !{i32 990000, i64 100, i32 1}
!15 = !{i32 999999, i64 1, i32 2}
