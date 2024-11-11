; RUN: opt -S -passes=gvn < %s | FileCheck %s

declare i32 @bar() #0

define i32 @foo() {
entry:
  %0 = tail call i32 @bar() #1
  %1 = tail call i32 @bar()
  ret i32 1
}

; CHECK-LABEL:    define i32 @foo(
; CHECK:            %0 = tail call i32 @bar() #1
; CHECK-NEXT:       %1 = tail call i32 @bar()
; CHECK-NEXT:       ret i32 1
; CHECK-NEXT:     }

attributes #0 = { memory(none) }
attributes #1 = { "llvm.assume"="ompx_no_call_asm" }
