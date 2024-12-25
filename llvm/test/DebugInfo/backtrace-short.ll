;; This test checks whether DWARF tag DW_AT_RUST_short_backtrace is accepted and processed.
; REQUIRES: object-emission
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump --verbose - | FileCheck %s
; CHECK: DW_AT_RUST_short_backtrace [DW_FORM_data1]

; ModuleID = 'backtrace.3cd23e1958f5234f-cgu.0'
source_filename = "backtrace.3cd23e1958f5234f-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; backtrace::foo
; Function Attrs: nonlazybind uwtable
define void @_ZN9backtrace3foo17h79548ad6a76bdf6cE() unnamed_addr #0 !dbg !7 {
start:
  ret void, !dbg !12
}

attributes #0 = { nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.dbg.cu = !{!5}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"RtLibUseGOT", i32 1}
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"rustc version 1.85.0-dev (\1B[0;95mlove you love you love you\1B[0m)"}
!5 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !6, producer: "clang LLVM (rustc version 1.85.0-dev (\1B[0;95mlove you love you love you\1B[0m))", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "backtrace.rs/@/backtrace.3cd23e1958f5234f-cgu.0", directory: "/home/jyn/src/example")
!7 = distinct !DISubprogram(name: "foo", linkageName: "_ZN9backtrace3foo17h79548ad6a76bdf6cE", scope: !9, file: !8, line: 4, type: !10, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, shortBacktrace: 0, unit: !5, templateParams: !11)
!8 = !DIFile(filename: "backtrace.rs", directory: "/home/jyn/src/example", checksumkind: CSK_MD5, checksum: "a2cf24d26a3fc1a5d0b6d8df3a534692")
!9 = !DINamespace(name: "backtrace", scope: null)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 4, column: 16, scope: !7)
