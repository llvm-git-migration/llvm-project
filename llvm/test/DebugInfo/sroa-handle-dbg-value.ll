; This test was obtained from swift source code and then automatically reducing it via Delta.
; The swift source code was from the test test/DebugInfo/debug_scope_distinct.swift.

; RUN: opt %s -S -p=sroa -o - | FileCheck %s

; CHECK: [[SROA_5_SROA_21:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_21]], metadata [[META59:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 72, 56)), !dbg [[DBG72:![0-9]+]]

; CHECK: tail call void @llvm.dbg.value(metadata ptr [[REG2:%[0-9]+]], metadata [[META54:![0-9]+]], metadata !DIExpression(DW_OP_deref)), !dbg [[DBG78:![0-9]+]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[REG2:%[0-9]+]], metadata [[META56:![0-9]+]], metadata !DIExpression(DW_OP_deref)), !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i64 0, metadata [[META57:![0-9]+]], metadata !DIExpression()), !dbg [[DBG78]]

; CHECK: [[SROA_418_SROA_COPYLOAD:%.*]] = load i8, ptr [[SROA_418_0_U1_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_418_SROA_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG72]]

%T4main1TV13TangentVectorV = type <{ %T4main1UV13TangentVectorV, [7 x i8], %T4main1UV13TangentVectorV }>
%T4main1UV13TangentVectorV = type <{ %T1M1SVySfG, [7 x i8], %T4main1VV13TangentVectorV }>
%T1M1SVySfG = type <{ ptr, %Ts4Int8V }>
%Ts4Int8V = type <{ i8 }>
%T4main1VV13TangentVectorV = type <{ %T1M1SVySfG }>
define hidden swiftcc void @"$s4main1TV13TangentVectorV1poiyA2E_AEtFZ"(ptr noalias nocapture sret(%T4main1TV13TangentVectorV) %0, ptr noalias nocapture dereferenceable(57) %1, ptr noalias nocapture dereferenceable(57) %2) #0 !dbg !44 {
  %7 = alloca %T4main1VV13TangentVectorV
  %8 = alloca %T4main1UV13TangentVectorV
  tail call void @llvm.dbg.value(metadata ptr %8, metadata !82, metadata !DIExpression()), !dbg !92
  tail call void @llvm.dbg.value(metadata ptr %1, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !95
  tail call void @llvm.dbg.value(metadata ptr %2, metadata !56, metadata !DIExpression(DW_OP_deref)), !dbg !95
  tail call void @llvm.dbg.value(metadata i64 0, metadata !57, metadata !DIExpression()), !dbg !95
  %.u2 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %1, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %8, ptr align 8 %.u2, i64 25, i1 false), !dbg !95
  %.s7 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %8, i32 0, i32 0
  %.s7.b = getelementptr inbounds %T1M1SVySfG, ptr %.s7, i32 0, i32 1
  %.s7.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s7.b, i32 0, i32 0
  %26 = load i8, ptr %.s7.b._value
  %.v9 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %8, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %.v9, i64 9, i1 false)
  %.s11 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %7, i32 0, i32 0
  %.s11.c = getelementptr inbounds %T1M1SVySfG, ptr %.s11, i32 0, i32 0
  %32 = load ptr, ptr %.s11.c
  ret void
}
!llvm.module.flags = !{ !7, !15}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"Swift Minor Version", i8 0}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, sdk: "MacOSX14.2.sdk")
!17 = !DIFile(filename: "/Users/shubham/Development/apple/swift/test/IRGen/debug_scope_distinct.swift", directory: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/test")
!44 = distinct !DISubprogram( unit: !16, retainedNodes: !53)
!53 = !{}
!54 = !DILocalVariable( scope: !44, flags: DIFlagArtificial)
!56 = !DILocalVariable( scope: !44, flags: DIFlagArtificial)
!57 = !DILocalVariable( scope: !44, flags: DIFlagArtificial)
!74 = distinct !DISubprogram( unit: !16, retainedNodes: !81)
!81 = !{}
!82 = !DILocalVariable( scope: !74, flags: DIFlagArtificial)
!91 = distinct !DILocation( scope: !44)
!92 = !DILocation( scope: !74, inlinedAt: !91)
!95 = !DILocation( scope: !44)
