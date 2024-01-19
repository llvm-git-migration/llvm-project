;; Tests memprof when contains recursion.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; TODO: Use text profile inputs once that is available for memprof.
;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.
;; # To generate below LLVM IR for use in matching.
;; $ clang++ -gmlt -fdebug-info-for-profiling -S %S/Inputs/memprof_direct_recursion_b.cc -emit-llvm

; RUN: llvm-profdata merge %S/Inputs/memprof_direct_recursion.memprofraw --profiled-binary %S/Inputs/memprof_direct_recursion.exe -o %t.memprofdata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -S | FileCheck %s

; CHECK: !16 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
; CHECK: !20 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
; CHECK: !24 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
; CHECK: !28 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
; CHECK: !32 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}

; ModuleID = 'llvm/test/Transforms/PGOProfile/Inputs/memprof_direct_recursion_b.cc'
source_filename = "llvm/test/Transforms/PGOProfile/Inputs/memprof_direct_recursion_b.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global ptr, align 8
@b = external global i32, align 4

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z3fooi(i32 noundef %c) #0 !dbg !10 {
entry:
  %c.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %c, ptr %c.addr, align 4
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 4) #2, !dbg !13
  store ptr %call, ptr @a, align 8, !dbg !14
  %0 = load i32, ptr %c.addr, align 4, !dbg !15
  %and = and i32 %0, 1, !dbg !16
  %tobool = icmp ne i32 %and, 0, !dbg !15
  br i1 %tobool, label %if.then, label %if.end, !dbg !15

if.then:                                          ; preds = %entry
  store i32 0, ptr %i, align 4, !dbg !17
  br label %for.cond, !dbg !18

for.cond:                                         ; preds = %for.inc, %if.then
  %1 = load i32, ptr %i, align 4, !dbg !19
  %cmp = icmp slt i32 %1, 100, !dbg !21
  br i1 %cmp, label %for.body, label %for.end, !dbg !22

for.body:                                         ; preds = %for.cond
  %2 = load ptr, ptr @a, align 8, !dbg !23
  %arrayidx = getelementptr inbounds i32, ptr %2, i64 0, !dbg !23
  store i32 1, ptr %arrayidx, align 4, !dbg !24
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4, !dbg !25
  %inc = add nsw i32 %3, 1, !dbg !25
  store i32 %inc, ptr %i, align 4, !dbg !25
  br label %for.cond, !dbg !27, !llvm.loop !28

for.end:                                          ; preds = %for.cond
  br label %if.end, !dbg !32

if.end:                                           ; preds = %for.end, %entry
  %4 = load i32, ptr @b, align 4, !dbg !33
  %dec = add nsw i32 %4, -1, !dbg !33
  store i32 %dec, ptr @b, align 4, !dbg !33
  %5 = load i32, ptr @b, align 4, !dbg !34
  %tobool1 = icmp ne i32 %5, 0, !dbg !34
  br i1 %tobool1, label %if.then2, label %if.end3, !dbg !34

if.then2:                                         ; preds = %if.end
  %6 = load i32, ptr %c.addr, align 4, !dbg !35
  call void @_Z3fooi(i32 noundef %6), !dbg !36
  br label %if.end3, !dbg !37

if.end3:                                          ; preds = %if.then2, %if.end
  ret void, !dbg !38
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #1

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0git (https://github.com/llvm/llvm-project.git 480cc413b7f7e73f90646e5feeb598e36e4e9565)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "llvm/test/Transforms/PGOProfile/Inputs/memprof_direct_recursion_b.cc", directory: "/", checksumkind: CSK_MD5, checksum: "dfaa17f2cd48c9a0a44fa520085af8a0")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0git (https://github.com/llvm/llvm-project.git 480cc413b7f7e73f90646e5feeb598e36e4e9565)"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 9, scope: !10)
!14 = !DILocation(line: 5, column: 7, scope: !10)
!15 = !DILocation(line: 6, column: 9, scope: !10)
!16 = !DILocation(line: 6, column: 10, scope: !10)
!17 = !DILocation(line: 7, column: 18, scope: !10)
!18 = !DILocation(line: 7, column: 14, scope: !10)
!19 = !DILocation(line: 7, column: 25, scope: !20)
!20 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 2)
!21 = !DILocation(line: 7, column: 27, scope: !20)
!22 = !DILocation(line: 7, column: 9, scope: !20)
!23 = !DILocation(line: 8, column: 9, scope: !10)
!24 = !DILocation(line: 8, column: 14, scope: !10)
!25 = !DILocation(line: 7, column: 34, scope: !26)
!26 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 4)
!27 = !DILocation(line: 7, column: 9, scope: !26)
!28 = distinct !{!28, !29, !30, !31}
!29 = !DILocation(line: 7, column: 9, scope: !10)
!30 = !DILocation(line: 8, column: 16, scope: !10)
!31 = !{!"llvm.loop.mustprogress"}
!32 = !DILocation(line: 9, column: 5, scope: !10)
!33 = !DILocation(line: 10, column: 5, scope: !10)
!34 = !DILocation(line: 11, column: 9, scope: !10)
!35 = !DILocation(line: 12, column: 13, scope: !10)
!36 = !DILocation(line: 12, column: 9, scope: !10)
!37 = !DILocation(line: 13, column: 5, scope: !10)
!38 = !DILocation(line: 14, column: 1, scope: !10)