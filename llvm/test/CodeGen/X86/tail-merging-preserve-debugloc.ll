; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-tail-merge=1 -stop-after=branch-folder | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main(i1 %0) {
entry:
  br i1 %0, label %1, label %2

1:                                                ; preds = %entry
  store i64 1, ptr null, align 1
; CHECK: JMP_1 %bb.3, debug-location !3
  br label %3, !dbg !3

2:                                                ; preds = %entry
  store i64 0, ptr null, align 1
  br label %3

3:                                                ; preds = %2, %1
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2d07c91bb9d9c2fa4eee31a1aeed20e3")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocation(line: 17, column: 3, scope: !4)
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
