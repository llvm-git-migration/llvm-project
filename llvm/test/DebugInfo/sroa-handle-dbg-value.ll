; This test was obtained from swift source code and then automatically reducing it via Delta.
; The swift source code was from the test test/DebugInfo/debug_scope_distinct.swift.

; RUN: opt %s -S -p=sroa -o - | FileCheck %s
; CHECK: [[SROA_5_SROA_21:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_21]], metadata [[META59:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 72, 56)), !dbg [[DBG72:![0-9]+]]
; CHECK-NEXT: [[SROA_5_SROA_14:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_14]], metadata [[META68:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 72, 56)), !dbg [[DBG72]]
; CHECK-NEXT: [[SROA_5_SROA_7:%.*]] = alloca [7 x i8], align 8
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_7]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 72, 56)), !dbg [[DBG74:![0-9]+]]
; CHECK-NEXT: [[SROA_5_SROA_0:%.*]] = alloca [7 x i8], align 8

; CHECK: tail call void @llvm.dbg.value(metadata ptr [[REG2:%[0-9]+]], metadata [[META54:![0-9]+]], metadata !DIExpression(DW_OP_deref)), !dbg [[DBG78:![0-9]+]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[REG2:%[0-9]+]], metadata [[META56:![0-9]+]], metadata !DIExpression(DW_OP_deref)), !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i64 0, metadata [[META57:![0-9]+]], metadata !DIExpression()), !dbg [[DBG78]]

; CHECK: [[SROA_418_SROA_COPYLOAD:%.*]] = load i8, ptr [[SROA_418_0_U1_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_418_SROA_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG72]]

; CHECK: [[SROA_5_SROA_322_COPYLOAD:%.*]] = load ptr, ptr [[SROA_5_322_SROA_5_U1_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_322_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64)), !dbg [[DBG72]]

; CHECK: [[SROA_5_SROA_423_COPYLOAD:%.*]] = load i8, ptr [[SROA_5_423_SROA_5_U1_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_423_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 192, 8)), !dbg [[DBG72]]

; CHECK: [[SROA_10_0_COPYLOAD:%.*]] = load ptr, ptr [[U11:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_10_0_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG72]]

; CHECK: [[SROA_411_0_COPYLOAD:%.*]] = load i8, ptr [[SROA_411_U11_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_411_0_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG72]]

; CHECK: [[SROA_5_SROA_315_COPYLOAD:%.*]] = load ptr, ptr [[SROA_5_SROA_311_U1_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_315_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64)), !dbg [[DBG72]]

; CHECK: [[SROA_5_SROA_416_COPYLOAD:%.*]] = load i8, ptr [[SROA_5_SROA_411_U11_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_416_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 192, 8)), !dbg [[DBG72]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_322_COPYLOAD]], metadata [[META79:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG92:![0-9]+]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_423_COPYLOAD]], metadata [[META79]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG92]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_315_COPYLOAD]], metadata [[META88:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG92]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_416_COPYLOAD]], metadata [[META88]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG92]]

; CHECK: [[SROA_3_0_COPYLOAD:%.*]] = load ptr, ptr [[U2:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_3_0_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG74]]

; CHECK: [[SROA_44_0_COPYLOAD:%.*]] = load i8, ptr [[SROA_44_U2_IDX:%.*]] align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_44_0_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG74]]

; CHECK: [[SROA_5_SROA_38_COPYLOAD:%.*]] = load ptr, ptr [[SROA_5_SROA_38_U2_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_38_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64)), !dbg [[DBG74]]

; CHECK: [[SROA_5_SROA_49_COPYLOAD:%.*]] = load i8, ptr [[SROA_5_SROA_49_U2_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_49_COPYLOAD]], metadata [[META59]], metadata !DIExpression(DW_OP_LLVM_fragment, 192, 8)), !dbg [[DBG74]]

; CHECK: [[SROA_0_0_COPYLOAD:%.*]] = load ptr, ptr [[U26:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_0_0_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG74]]

; CHECK: [[SROA_4_0_COPYLOAD:%.*]] = load i8, ptr [[SROA_4_0_IDX:%.*]], align 8, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_4_0_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG74]]

; CHECK: [[SROA_5_SROA_3_COPYLOAD:%.*]] = load ptr, ptr [[SROA_5_SROA_3_SROA_5_U26_IDX_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_3_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 128, 64)), !dbg [[DBG74]]

; CHECK: [[SROA_5_SROA_4_COPYLOAD:%.*]] = load i8, ptr [[SROA_5_SROA_4_SROA_5_U26_IDX:%.*]], align 1, !dbg [[DBG78]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_4_COPYLOAD]], metadata [[META68]], metadata !DIExpression(DW_OP_LLVM_fragment, 192, 8)), !dbg [[DBG74]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_38_COPYLOAD]], metadata [[META79]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG94:![0-9]+]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_49_COPYLOAD]], metadata [[META79]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG94]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata ptr [[SROA_5_SROA_3_COPYLOAD]], metadata [[META88]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg [[DBG94]]
; CHECK-NEXT: tail call void @llvm.dbg.value(metadata i8 [[SROA_5_SROA_4_COPYLOAD]], metadata [[META88]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 8)), !dbg [[DBG94]]

%T4main1TV13TangentVectorV = type <{ %T4main1UV13TangentVectorV, [7 x i8], %T4main1UV13TangentVectorV }>
%T4main1UV13TangentVectorV = type <{ %T1M1SVySfG, [7 x i8], %T4main1VV13TangentVectorV }>
%T1M1SVySfG = type <{ ptr, %Ts4Int8V }>
%Ts4Int8V = type <{ i8 }>
%T4main1VV13TangentVectorV = type <{ %T1M1SVySfG }>
define hidden swiftcc void @"$s4main1TV13TangentVectorV1poiyA2E_AEtFZ"(ptr noalias nocapture sret(%T4main1TV13TangentVectorV) %0, ptr noalias nocapture dereferenceable(57) %1, ptr noalias nocapture dereferenceable(57) %2) #0 !dbg !44 {
entry:
  %3 = alloca %T4main1VV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %3, metadata !59, metadata !DIExpression()), !dbg !72
  %4 = alloca %T4main1UV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %4, metadata !82, metadata !DIExpression()), !dbg !88
  %5 = alloca %T4main1VV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %5, metadata !68, metadata !DIExpression()), !dbg !72
  %6 = alloca %T4main1UV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %6, metadata !84, metadata !DIExpression()), !dbg !88
  %7 = alloca %T4main1VV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %7, metadata !59, metadata !DIExpression()), !dbg !89
  %8 = alloca %T4main1UV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %8, metadata !82, metadata !DIExpression()), !dbg !92
  %9 = alloca %T4main1VV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %9, metadata !68, metadata !DIExpression()), !dbg !89
  %10 = alloca %T4main1UV13TangentVectorV, align 8
  tail call void @llvm.dbg.value(metadata ptr %10, metadata !84, metadata !DIExpression()), !dbg !92
  call void @llvm.lifetime.start.p0(i64 9, ptr %3)
  call void @llvm.lifetime.start.p0(i64 25, ptr %4)
  call void @llvm.lifetime.start.p0(i64 9, ptr %5)
  call void @llvm.lifetime.start.p0(i64 25, ptr %6), !dbg !93
  call void @llvm.lifetime.start.p0(i64 9, ptr %7)
  call void @llvm.lifetime.start.p0(i64 25, ptr %8)
  call void @llvm.lifetime.start.p0(i64 9, ptr %9)
  call void @llvm.lifetime.start.p0(i64 25, ptr %10)
  tail call void @llvm.dbg.value(metadata ptr %1, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !95
  tail call void @llvm.dbg.value(metadata ptr %2, metadata !56, metadata !DIExpression(DW_OP_deref)), !dbg !95
  tail call void @llvm.dbg.value(metadata i64 0, metadata !57, metadata !DIExpression()), !dbg !95
  %.u1 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %1, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %4, ptr align 8 %.u1, i64 25, i1 false), !dbg !95
  %.u11 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %2, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %6, ptr align 8 %.u11, i64 25, i1 false), !dbg !95
  %.s = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 0
  %.s.c = getelementptr inbounds %T1M1SVySfG, ptr %.s, i32 0, i32 0
  %11 = load ptr, ptr %.s.c, align 8
  %.s.b = getelementptr inbounds %T1M1SVySfG, ptr %.s, i32 0, i32 1
  %.s.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s.b, i32 0, i32 0
  %12 = load i8, ptr %.s.b._value, align 8
  %.s2 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %6, i32 0, i32 0
  %.s2.c = getelementptr inbounds %T1M1SVySfG, ptr %.s2, i32 0, i32 0
  %13 = load ptr, ptr %.s2.c, align 8
  %.s2.b = getelementptr inbounds %T1M1SVySfG, ptr %.s2, i32 0, i32 1
  %.s2.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s2.b, i32 0, i32 0
  %14 = load i8, ptr %.s2.b._value, align 8
  %.v = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %4, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %3, ptr align 8 %.v, i64 9, i1 false)
  %.v3 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %6, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %5, ptr align 8 %.v3, i64 9, i1 false)
  %.s4 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %3, i32 0, i32 0
  %.s4.c = getelementptr inbounds %T1M1SVySfG, ptr %.s4, i32 0, i32 0
  %18 = load ptr, ptr %.s4.c, align 8
  %.s5 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %5, i32 0, i32 0
  %.s5.c = getelementptr inbounds %T1M1SVySfG, ptr %.s5, i32 0, i32 0
  %20 = load ptr, ptr %.s5.c, align 8
  %.u2 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %1, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %8, ptr align 8 %.u2, i64 25, i1 false), !dbg !95
  %.u26 = getelementptr inbounds %T4main1TV13TangentVectorV, ptr %2, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %10, ptr align 8 %.u26, i64 25, i1 false), !dbg !95
  %.s7 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %8, i32 0, i32 0
  %.s7.c = getelementptr inbounds %T1M1SVySfG, ptr %.s7, i32 0, i32 0
  %25 = load ptr, ptr %.s7.c, align 8
  %.s7.b = getelementptr inbounds %T1M1SVySfG, ptr %.s7, i32 0, i32 1
  %.s7.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s7.b, i32 0, i32 0
  %26 = load i8, ptr %.s7.b._value, align 8
  %.s8 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %10, i32 0, i32 0
  %.s8.c = getelementptr inbounds %T1M1SVySfG, ptr %.s8, i32 0, i32 0
  %27 = load ptr, ptr %.s8.c, align 8
  %.s8.b = getelementptr inbounds %T1M1SVySfG, ptr %.s8, i32 0, i32 1
  %.s8.b._value = getelementptr inbounds %Ts4Int8V, ptr %.s8.b, i32 0, i32 0
  %28 = load i8, ptr %.s8.b._value, align 8
  %.v9 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %8, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %.v9, i64 9, i1 false)
  %.v10 = getelementptr inbounds %T4main1UV13TangentVectorV, ptr %10, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %9, ptr align 8 %.v10, i64 9, i1 false)
  %.s11 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %7, i32 0, i32 0
  %.s11.c = getelementptr inbounds %T1M1SVySfG, ptr %.s11, i32 0, i32 0
  %32 = load ptr, ptr %.s11.c, align 8
  %.s12 = getelementptr inbounds %T4main1VV13TangentVectorV, ptr %9, i32 0, i32 0
  %.s12.c = getelementptr inbounds %T1M1SVySfG, ptr %.s12, i32 0, i32 0
  %34 = load ptr, ptr %.s12.c, align 8
  call void @llvm.lifetime.end.p0(i64 25, ptr %10)
  call void @llvm.lifetime.end.p0(i64 9, ptr %9)
  call void @llvm.lifetime.end.p0(i64 25, ptr %8)
  call void @llvm.lifetime.end.p0(i64 9, ptr %7)
  call void @llvm.lifetime.end.p0(i64 25, ptr %6)
  call void @llvm.lifetime.end.p0(i64 9, ptr %5)
  call void @llvm.lifetime.end.p0(i64 25, ptr %4)
  call void @llvm.lifetime.end.p0(i64 9, ptr %3)
  ret void
}
!llvm.module.flags = !{!0, !1, !2, !3, !4, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!swift.module.flags = !{!33}
!llvm.linker.options = !{!34, !35, !36, !37, !38, !39, !40, !41, !42, !43}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 2]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{i32 7, !"frame-pointer", i32 1}
!12 = !{i32 1, !"Swift Version", i32 7}
!13 = !{i32 1, !"Swift ABI Version", i32 7}
!14 = !{i32 1, !"Swift Major Version", i8 6}
!15 = !{i32 1, !"Swift Minor Version", i8 0}
!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, imports: !18, sdk: "MacOSX14.2.sdk")
!17 = !DIFile(filename: "/Users/shubham/Development/apple/swift/test/IRGen/debug_scope_distinct.swift", directory: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/test")
!18 = !{!19, !21, !23, !25, !27, !29, !31}
!19 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !20, file: !17)
!20 = !DIModule(scope: null, name: "main", includePath: "/Users/shubham/Development/apple/swift/test/IRGen")
!21 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !22, file: !17)
!22 = !DIModule(scope: null, name: "Swift", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule")
!23 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !24, line: 60)
!24 = !DIModule(scope: null, name: "_Differentiation", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/lib/swift/macosx/_Differentiation.swiftmodule/arm64-apple-macos.swiftmodule")
!25 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !26, line: 61)
!26 = !DIModule(scope: null, name: "M", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/test-macosx-arm64/IRGen/Output/debug_scope_distinct.swift.tmp/M.swiftmodule")
!27 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !28, file: !17)
!28 = !DIModule(scope: null, name: "_StringProcessing", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/lib/swift/macosx/_StringProcessing.swiftmodule/arm64-apple-macos.swiftmodule")
!29 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !30, file: !17)
!30 = !DIModule(scope: null, name: "_SwiftConcurrencyShims", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/lib/swift/shims")
!31 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !17, entity: !32, file: !17)
!32 = !DIModule(scope: null, name: "_Concurrency", includePath: "/Users/shubham/Development/apple/build/Ninja-RelWithDebInfoAssert/swift-macosx-arm64/lib/swift/macosx/_Concurrency.swiftmodule/arm64-apple-macos.swiftmodule")
!33 = !{i1 false}
!34 = !{!"-lswiftCore"}
!35 = !{!"-lswift_StringProcessing"}
!36 = !{!"-lswift_Differentiation"}
!37 = !{!"-lswiftDarwin"}
!38 = !{!"-lswift_Concurrency"}
!39 = !{!"-lswiftSwiftOnoneSupport"}
!40 = !{!"-lobjc"}
!41 = !{!"-lswiftCompatibilityConcurrency"}
!42 = !{!"-lswiftCompatibility56"}
!43 = !{!"-lswiftCompatibilityPacks"}
!44 = distinct !DISubprogram(scope: !46, file: !45, type: !49, unit: !16, declaration: !52, retainedNodes: !53)
!45 = !DIFile(filename: "<compiler-generated>", directory: "/")
!46 = !DICompositeType(tag: DW_TAG_structure_type, scope: !47, elements: !48, identifier: "$s4main1TV13TangentVectorVD")
!47 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1TVD")
!48 = !{}
!49 = !DISubroutineType(types: !50)
!50 = !{!51}
!51 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1TV13TangentVectorVXMtD")
!52 = !DISubprogram(spFlags: DISPFlagOptimized)
!53 = !{!54, !56, !57}
!54 = !DILocalVariable(scope: !44, type: !55, flags: DIFlagArtificial)
!55 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !46)
!56 = !DILocalVariable(scope: !44, flags: DIFlagArtificial)
!57 = !DILocalVariable(scope: !44, type: !58, flags: DIFlagArtificial)
!58 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !51)
!59 = !DILocalVariable(scope: !60, type: !69, flags: DIFlagArtificial)
!60 = distinct !DISubprogram(unit: !16, declaration: !66, retainedNodes: !67)
!61 = !DICompositeType(tag: DW_TAG_structure_type, scope: !62, identifier: "$s4main1VV13TangentVectorVD")
!62 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1VVD")
!63 = !DISubroutineType(types: !64)
!64 = !{!61, !65}
!65 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1VV13TangentVectorVXMtD")
!66 = !DISubprogram(type: !63, spFlags: DISPFlagOptimized)
!67 = !{!68, !70}
!68 = !DILocalVariable(scope: !60, flags: DIFlagArtificial)
!69 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !61)
!70 = !DILocalVariable(scope: !60, type: !71, flags: DIFlagArtificial)
!71 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !65)
!72 = !DILocation(scope: !60, inlinedAt: !73)
!73 = distinct !DILocation(scope: !74, inlinedAt: !87)
!74 = distinct !DISubprogram(type: !77, unit: !16, declaration: !80, retainedNodes: !81)
!75 = !DICompositeType(tag: DW_TAG_structure_type, scope: !76, identifier: "$s4main1UV13TangentVectorVD")
!76 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1UVD")
!77 = !DISubroutineType(types: !78)
!78 = !{!75, !79}
!79 = !DICompositeType(tag: DW_TAG_structure_type, identifier: "$s4main1UV13TangentVectorVXMtD")
!80 = !DISubprogram(name: "+", spFlags: DISPFlagOptimized)
!81 = !{!84, !85}
!82 = !DILocalVariable(scope: !74, type: !83, flags: DIFlagArtificial)
!83 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !75)
!84 = !DILocalVariable(scope: !74, flags: DIFlagArtificial)
!85 = !DILocalVariable(scope: !74, type: !86, flags: DIFlagArtificial)
!86 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !79)
!87 = distinct !DILocation(scope: !44)
!88 = !DILocation(scope: !74, inlinedAt: !87)
!89 = !DILocation(scope: !60, inlinedAt: !90)
!90 = distinct !DILocation(scope: !74, inlinedAt: !91)
!91 = distinct !DILocation(scope: !44)
!92 = !DILocation(scope: !74, inlinedAt: !91)
!93 = !DILocation(scope: !94)
!94 = !DILexicalBlockFile(scope: !44, discriminator: 0)
!95 = !DILocation(scope: !44)
