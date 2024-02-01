; RUN: split-file %s %t
; RUN: llc -filetype=obj --mtriple=wasm32-unknown-unknown -o %t/main.o %t/main.ll
; RUN: llc -filetype=obj --mtriple=wasm32-unknown-unknown -o %t/liba_x.o %t/liba_x.ll
; RUN: llc -filetype=obj --mtriple=wasm32-unknown-unknown -o %t/liba_y.o %t/liba_y.ll
; RUN: llvm-ar rcs %t/liba.a %t/liba_x.o %t/liba_y.o
; RUN: wasm-ld %t/main.o %t/liba.a --gc-sections -o %t/main.wasm --print-gc-sections | FileCheck %s --check-prefix=GC
; RUN: obj2yaml %t/main.wasm | FileCheck %s

; --gc-sections should remove non-retained and unused "weathers" section from live object liba_x.o
; GC: removing unused section {{.*}}/liba.a(liba_x.o):(weathers)
; Should not remove retained "greetings" sections from live objects main.o and liba_x.o
; GC-NOT: removing unused section %t/main.o:(greetings)
; GC-NOT: removing unused section %t/liba_x.o:(greetings)

; Note: All symbols are private so that they don't join the symbol table.

;--- main.ll

@greet_a = private constant [6 x i8] c"hello\00", align 1, section "greetings"
@weather_a = private constant [7 x i8] c"cloudy\00", align 1, section "weathers"
@llvm.used = appending global [2 x ptr] [ptr @greet_a, ptr @weather_a], section "llvm.metadata"

declare void @grab_liba()
define void @_start() {
  call void @grab_liba()
  ret void
}

;--- liba_x.ll

@greet_b = private constant [6 x i8] c"world\00", align 1, section "greetings"
@weather_b = private constant [6 x i8] c"rainy\00", align 1, section "weathers"

@llvm.used = appending global [1 x ptr] [ptr @greet_b], section "llvm.metadata"

define void @grab_liba() {
  ret void
}

;--- liba_y.ll
@greet_d = private constant [4 x i8] c"bye\00", align 1, section "greetings"

@llvm.used = appending global [1 x ptr] [ptr @greet_d], section "llvm.metadata"


; "greetings" section
; CHECK: - Type:            DATA
; CHECK:   Segments:
; CHECK:     - SectionOffset:   7
; CHECK:       InitFlags:       0
; CHECK:       Offset:
; CHECK:         Opcode:          I32_CONST
; CHECK:         Value:           1024
; CHECK:       Content:         68656C6C6F00776F726C6400
; "weahters" section.
; CHECK: - SectionOffset:   25
; CHECK:   InitFlags:       0
; CHECK:   Offset:
; CHECK:     Opcode:          I32_CONST
; CHECK:     Value:           1036
; CHECK:   Content:         636C6F75647900
