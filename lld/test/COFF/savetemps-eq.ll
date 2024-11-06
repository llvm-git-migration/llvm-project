; REQUIRES: x86
; RUN: rm -fr %T/savetemps-eq
; RUN: mkdir %T/savetemps-eq
; RUN: opt -thinlto-bc -o %T/savetemps-eq/savetemps.obj %s
; RUN: opt -thinlto-bc -o %T/savetemps-eq/thin1.obj %S/Inputs/thinlto.ll

;; Check preopt
; RUN: lld-link /lldsavetemps:preopt /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.obj.*.preopt.bc | count 2

;; Check promote
; RUN: lld-link /lldsavetemps:promote /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.obj.*.promote.bc | count 2

;; Check internalize
; RUN: lld-link /lldsavetemps:internalize /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.obj.*.internalize.bc | count 2

;; Check import
; RUN: lld-link /lldsavetemps:import /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.obj.*.import.bc | count 2

;; Check opt
;; Not supported on Windows due to difficulty with escaping "opt" across platforms.

;; Check precodegen
; RUN: lld-link /lldsavetemps:precodegen /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.obj.*.precodegen.bc | count 2

;; Check combinedindex
; RUN: lld-link /lldsavetemps:combinedindex /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.exe.index.bc | count 1

;; Check prelink
; RUN: lld-link /lldsavetemps:prelink /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.exe.lto.*.obj | count 2

;; Check resolution
; RUN: lld-link /lldsavetemps:resolution /out:%T/savetemps-eq/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-eq/savetemps.obj %T/savetemps-eq/thin1.obj
; RUN: ls %T/savetemps-eq/*.resolution.txt | count 1

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @g()

define i32 @main() {
  call void @g()
  ret i32 0
}
