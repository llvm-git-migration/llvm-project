; RUN: llc --filetype=obj %s -o -
%"class.llvm::StringRef" = type { ptr, i64 }
define internal void @_ZL30tokenizeWindowsCommandLineImplN4llvm9StringRefERNS_11StringSaverENS_12function_refIFvS0_EEEbNS3_IFvvEEEb() #0 !dbg !12 {
  %7 = alloca %"class.llvm::StringRef", align 8
  %21 = call noundef i64 @_ZNK4llvm9StringRef4sizeEv(ptr noundef nonnull align 8 dereferenceable(16) %7), !dbg !264
  br label %22, !dbg !265
  br label %22, !llvm.loop !284
}
define linkonce_odr noundef i64 @_ZNK4llvm9StringRef4sizeEv() #0 align 2 !dbg !340 {
  %2 = alloca ptr, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.llvm::StringRef", ptr %3, !dbg !344
  %5 = load i64, ptr %4, !dbg !344
  ret i64 %5, !dbg !345
}
!llvm.module.flags = !{!2, !6}
!llvm.dbg.cu = !{!7}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, sdk: "MacOSX14.0.sdk")
!8 = !DIFile(filename: "file.cpp", directory: "/Users/Dev", checksumkind: CSK_MD5, checksum: "ed7ae158f20f7914bc5fb843291e80da")
!12 = distinct !DISubprogram(unit: !7, retainedNodes: !36)
!36 = !{}
!260 = distinct !DILexicalBlock(scope: !12, line: 412, column: 3)
!264 = !DILocation(scope: !260)
!265 = !DILocation(scope: !260, column: 20)
!284 = distinct !{}
!340 = distinct !DISubprogram(unit: !7, retainedNodes: !36)
!344 = !DILocation(scope: !340)
!345 = !DILocation(scope: !340)
