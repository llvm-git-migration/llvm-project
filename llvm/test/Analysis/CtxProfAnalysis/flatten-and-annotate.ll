; REQUIRES: x86_64-linux
;
; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   %t/example.ll -S -o %t/prelink.ll
; RUN: FileCheck --input-file %t/prelink.ll %s --check-prefix=PRELINK
; RUN: opt -passes='ctx-prof-flatten' -use-ctx-profile=%t/profile.ctxprofdata %t/prelink.ll -S  | FileCheck %s
;
;
; Check that instrumentation occurs where expected: the "no" block for foo, and
; the "yes" block for an_entrypoint - which explains the subsequent branch weights
;
; PRELINK-LABEL: @foo
; PRELINK-LABEL: no:
; PRELINK:         call void @llvm.instrprof.increment(ptr @foo, i64 [[#]], i32 2, i32 1)

; PRELINK-LABEL: @an_entrypoint
; PRELINK-LABEL: yes:
; PRELINK:         call void @llvm.instrprof.increment(ptr @an_entrypoint, i64 [[#]], i32 2, i32 1)

; Check that the output has:
;  - no instrumentation
;  - the 2 functions have an entry count
;  - each conditional branch has profile annotation
;
; CHECK-NOT:   call void @llvm.instrprof
;
; CHECK-LABEL: @foo
; CHECK-SAME:    !prof !0
; CHECK:          br i1 %t, label %yes, label %no, !prof !2
; CHECK-LABEL: @an_entrypoint
; CHECK-SAME:    !prof !3
; CHECK:          br i1 %t, label %yes, label %common.ret, !prof !5
; CHECK:       !0 = !{!"function_entry_count", i64 40} 
; CHECK:       !2 = !{!"branch_weights", i32 30, i32 10} 
; CHECK:       !5 = !{!"branch_weights", i32 40, i32 60} 

;--- profile.json
[
  {
    "Guid": 4909520559318251808,
    "Counters": [100, 40],
    "Callsites": [
      [
        {
          "Guid": 11872291593386833696,
          "Counters": [ 40, 10 ]
        }
      ]
    ]
  }
]
;--- example.ll
declare void @bar()

define void @foo(i32 %a, ptr %fct) #0 !guid !0 {
  %t = icmp sgt i32 %a, 7
  br i1 %t, label %yes, label %no
yes:
  call void %fct(i32 %a)
  br label %exit
no:
  call void @bar()
  br label %exit
exit:
  ret void
}

define void @an_entrypoint(i32 %a) !guid !1 {
  %t = icmp sgt i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  call void @foo(i32 1, ptr null)
  ret void
no:
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 11872291593386833696 }
!1 = !{i64 4909520559318251808}
