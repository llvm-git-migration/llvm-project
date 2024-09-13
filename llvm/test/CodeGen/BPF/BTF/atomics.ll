; RUN: llc -march=bpfel -mcpu=v3 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -mcpu=v3 -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #include <stdatomic.h>
;   struct gstruct_t {
;     _Atomic int a;
;   } gstruct;
;   extern _Atomic int ext;
;   _Atomic int gbl;
;   _Atomic int *pgbl;
;   volatile _Atomic int vvar;
;   _Atomic int foo(_Atomic int a1, _Atomic int *p1) {
;     (void)__c11_atomic_fetch_add(&gstruct.a, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&ext, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&gbl, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(pgbl, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(&vvar, 1, memory_order_relaxed);
;     (void)__c11_atomic_fetch_add(p1, 1, memory_order_relaxed);
;
;     return a1;
;   }

target triple = "bpf"

%struct.gstruct_t = type { i32 }

@gstruct = dso_local global %struct.gstruct_t zeroinitializer, align 4, !dbg !0
@ext = external dso_local global i32, align 4, !dbg !26
@gbl = dso_local global i32 0, align 4, !dbg !16
@pgbl = dso_local local_unnamed_addr global ptr null, align 8, !dbg !20
@vvar = dso_local global i32 0, align 4, !dbg !23

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define dso_local i32 @foo(i32 returned %a1, ptr nocapture noundef %p1) local_unnamed_addr #0 !dbg !37 {
entry:
    #dbg_value(i32 %a1, !41, !DIExpression(), !43)
    #dbg_value(ptr %p1, !42, !DIExpression(), !43)
  %0 = atomicrmw add ptr @gstruct, i32 1 monotonic, align 4, !dbg !44
  %1 = atomicrmw add ptr @ext, i32 1 monotonic, align 4, !dbg !45
  %2 = atomicrmw add ptr @gbl, i32 1 monotonic, align 4, !dbg !46
  %3 = load ptr, ptr @pgbl, align 8, !dbg !47, !tbaa !48
  %4 = atomicrmw add ptr %3, i32 1 monotonic, align 4, !dbg !52
  %5 = atomicrmw volatile add ptr @vvar, i32 1 monotonic, align 4, !dbg !53
  %6 = atomicrmw add ptr %p1, i32 1 monotonic, align 4, !dbg !54
  ret i32 %a1, !dbg !55
}

; CHECK:             .long   1                               # BTF_KIND_INT(id = 1)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT:        .long   218103810                       # 0xd000002
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   11                              # BTF_KIND_FUNC(id = 4)
; CHECK-NEXT:        .long   201326593                       # 0xc000001
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   66                              # BTF_KIND_STRUCT(id = 5)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   76
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # 0x0
; CHECK-NEXT:        .long   78                              # BTF_KIND_VAR(id = 6)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   86                              # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   90                              # BTF_KIND_VAR(id = 8)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   94                              # BTF_KIND_VAR(id = 9)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   0                               # BTF_KIND_VOLATILE(id = 10)
; CHECK-NEXT:        .long   150994944                       # 0x9000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   99                              # BTF_KIND_VAR(id = 11)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   104                             # BTF_KIND_DATASEC(id = 12)
; CHECK-NEXT:        .long   251658244                       # 0xf000004
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   gstruct
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   gbl
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   pgbl
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   11
; CHECK-NEXT:        .long   vvar
; CHECK-NEXT:        .long   4


; CHECK:             .byte   0                               # string offset=0
; CHECK:             .ascii  "int"                           # string offset=1
; CHECK:             .ascii  "a1"                            # string offset=5
; CHECK:             .ascii  "p1"                            # string offset=8
; CHECK:             .ascii  "foo"                           # string offset=11
; CHECK:             .ascii  "gstruct_t"                     # string offset=66
; CHECK:             .byte   97                              # string offset=76
; CHECK:             .ascii  "gstruct"                       # string offset=78
; CHECK:             .ascii  "ext"                           # string offset=86
; CHECK:             .ascii  "gbl"                           # string offset=90
; CHECK:             .ascii  "pgbl"                          # string offset=94
; CHECK:             .ascii  "vvar"                          # string offset=99

attributes #0 = { mustprogress nofree norecurse nounwind willreturn "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="v3" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!31, !32, !33, !34, !35}
!llvm.ident = !{!36}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gstruct", scope: !2, file: !3, line: 4, type: !28, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git a7bdb883df5731338d84603c60210d93c86f0870)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !15, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "atomics.c", directory: "/tmp/home/yhs/tests/result/atomics", checksumkind: CSK_MD5, checksum: "cabe3f3bfcfa90a93ff6d959be6e563a")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "memory_order", file: !6, line: 68, baseType: !7, size: 32, elements: !8)
!6 = !DIFile(filename: "work/yhs/llvm-project/llvm/build/install/lib/clang/20/include/stdatomic.h", directory: "/home/yhs", checksumkind: CSK_MD5, checksum: "f17199a988fe91afffaf0f943ef87096")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11, !12, !13, !14}
!9 = !DIEnumerator(name: "memory_order_relaxed", value: 0)
!10 = !DIEnumerator(name: "memory_order_consume", value: 1)
!11 = !DIEnumerator(name: "memory_order_acquire", value: 2)
!12 = !DIEnumerator(name: "memory_order_release", value: 3)
!13 = !DIEnumerator(name: "memory_order_acq_rel", value: 4)
!14 = !DIEnumerator(name: "memory_order_seq_cst", value: 5)
!15 = !{!0, !16, !20, !23, !26}
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "gbl", scope: !2, file: !3, line: 6, type: !18, isLocal: false, isDefinition: true)
!18 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !19)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "pgbl", scope: !2, file: !3, line: 7, type: !22, isLocal: false, isDefinition: true)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "vvar", scope: !2, file: !3, line: 8, type: !25, isLocal: false, isDefinition: true)
!25 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !18)
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression())
!27 = distinct !DIGlobalVariable(name: "ext", scope: !2, file: !3, line: 5, type: !18, isLocal: false, isDefinition: false)
!28 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "gstruct_t", file: !3, line: 2, size: 32, elements: !29)
!29 = !{!30}
!30 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !28, file: !3, line: 3, baseType: !18, size: 32)
!31 = !{i32 7, !"Dwarf Version", i32 5}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 1, !"wchar_size", i32 4}
!34 = !{i32 7, !"frame-pointer", i32 2}
!35 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!36 = !{!"clang version 20.0.0git (git@github.com:yonghong-song/llvm-project.git a7bdb883df5731338d84603c60210d93c86f0870)"}
!37 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 9, type: !38, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !40)
!38 = !DISubroutineType(types: !39)
!39 = !{!18, !18, !22}
!40 = !{!41, !42}
!41 = !DILocalVariable(name: "a1", arg: 1, scope: !37, file: !3, line: 9, type: !18)
!42 = !DILocalVariable(name: "p1", arg: 2, scope: !37, file: !3, line: 9, type: !22)
!43 = !DILocation(line: 0, scope: !37)
!44 = !DILocation(line: 10, column: 9, scope: !37)
!45 = !DILocation(line: 11, column: 9, scope: !37)
!46 = !DILocation(line: 12, column: 9, scope: !37)
!47 = !DILocation(line: 13, column: 32, scope: !37)
!48 = !{!49, !49, i64 0}
!49 = !{!"any pointer", !50, i64 0}
!50 = !{!"omnipotent char", !51, i64 0}
!51 = !{!"Simple C/C++ TBAA"}
!52 = !DILocation(line: 13, column: 9, scope: !37)
!53 = !DILocation(line: 14, column: 9, scope: !37)
!54 = !DILocation(line: 15, column: 9, scope: !37)
!55 = !DILocation(line: 17, column: 3, scope: !37)
