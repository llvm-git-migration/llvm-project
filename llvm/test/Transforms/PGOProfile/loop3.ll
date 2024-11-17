; RUN: opt %s -passes=pgo-instr-gen -pgo-instrument-loop-entries=false -S | FileCheck %s --check-prefixes=GEN,NOTLOOPENTRIES
; RUN: llvm-profdata merge %S/Inputs/loop3.proftext -o %t.profdata
; RUN: opt %s -passes=pgo-instr-use -pgo-instrument-loop-entries=false -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt %s -passes=pgo-instr-gen -pgo-instrument-loop-entries=true -S | FileCheck %s --check-prefixes=GEN,LOOPENTRIES
; RUN: llvm-profdata merge %S/Inputs/loop3_loop_entries.proftext -o %t.profdata
; RUN: opt %s -passes=pgo-instr-use -pgo-instrument-loop-entries=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt %s -passes=pgo-instr-gen -pgo-instrument-entry=true -S | FileCheck %s --check-prefixes=GEN,FUNCTIONENTRY
; RUN: llvm-profdata merge %S/Inputs/loop3_function_entry.proftext -o %t.profdata
; RUN: opt %s -passes=pgo-instr-use -pgo-instrument-entry=true -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

; GEN: $__llvm_profile_raw_version = comdat any
; GEN: @__llvm_profile_raw_version = hidden constant i64 {{[0-9]+}}, comdat
; GEN: @__profn_test_simple_for_with_bypass = private constant [27 x i8] c"test_simple_for_with_bypass"

define i32 @test_simple_for_with_bypass(i32 %n) {
; USE: define i32 @test_simple_for_with_bypass(i32 %n)
; USE-SAME: !prof ![[ENTRY_COUNT:[0-9]*]]
entry:
; GEN: entry:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
  %mask = and i32 %n, 65535
  %skip = icmp eq i32 %mask, 0
  br i1 %skip, label %end, label %for.entry
; USE: br i1 %skip, label %end, label %for.entry
; USE-SAME: !prof ![[BW_FOR_BYPASS:[0-9]+]]

for.entry:
; GEN: for.entry:
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
; NOTLOOPENTRIES-NOT: call void @llvm.instrprof.increment
; FUNCTIONENTRY-NOT: call void @llvm.instrprof.increment
  br label %for.cond

for.cond:
; GEN: for.cond:
; GEN-NOT: call void @llvm.instrprof.increment
  %i = phi i32 [ 0, %for.entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %for.entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end, !prof !1
; USE: br i1 %cmp, label %for.body, label %for.end
; USE-SAME: !prof ![[BW_FOR_COND:[0-9]+]]

for.body:
; GEN: for.body:
; GEN-NOT: call void @llvm.instrprof.increment
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
; GEN: for.inc:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 0)
; LOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 1)
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
; GEN: for.end:
; NOTLOOPENTRIES: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
; FUNCTIONENTRY: call void @llvm.instrprof.increment(ptr @__profn_test_simple_for_with_bypass, i64 {{[0-9]+}}, i32 3, i32 2)
; LOOPENTRIES-NOT: call void @llvm.instrprof.increment
  br label %end

end:
; GEN: end:
; GEN-NOT: call void @llvm.instrprof.increment
  %final_sum = phi i32 [ %sum, %for.end ], [ 0, %entry ]
  ret i32 %final_sum
}

!1 = !{!"branch_weights", i32 100000, i32 80}

; USE: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 12}
; USE: ![[BW_FOR_BYPASS]] = !{!"branch_weights", i32 4, i32 8}
; USE: ![[BW_FOR_COND]] = !{!"branch_weights", i32 123456, i32 8}
