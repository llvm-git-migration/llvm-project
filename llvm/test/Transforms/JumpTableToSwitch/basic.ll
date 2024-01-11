; RUN: opt < %s -passes=jump-table-to-switch -verify-dom-info -S | FileCheck %s
; RUN: opt < %s -passes=jump-table-to-switch -jump-table-to-switch-size-threshold=0 -verify-dom-info -S | FileCheck %s --check-prefix=THRESHOLD-0

@func_array = global [2 x ptr] [ptr @func0, ptr @func1]

define i32 @func0() {
  ret i32 1
}

define i32 @func1() {
  ret i32 2
}

define i32 @function_with_jump_table(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep
  %result = call i32 %func_ptr()
  ret i32 %result
}

; CHECK-LABEL: define i32 @function_with_jump_table
; CHECK:       [[GEP:%.*]] = getelementptr inbounds [2 x ptr], ptr @func_array, i32 0, i32 %index
; CHECK:       switch i32 %index, label %[[DEFAULT_LABEL:.*]] [
; CHECK-NEXT:    i32 0, label %[[CALL_0_LABEL:.*]]
; CHECK-NEXT:    i32 1, label %[[CALL_1_LABEL:.*]]
; CHECK-NEXT:  ]

; CHECK:       [[DEFAULT_LABEL]]
; CHECK-NEXT:    unreachable

; CHECK:       [[CALL_0_LABEL]]:
; CHECK-NEXT:    %1 = call i32 @func0()
; CHECK-NEXT:    br label %[[TAIL_LABEL:.*]]

; CHECK:       [[CALL_1_LABEL]]:
; CHECK-NEXT:    %2 = call i32 @func1()
; CHECK-NEXT:    br label %[[TAIL_LABEL]]

; CHECK:       [[TAIL_LABEL]]:
; CHECK-NEXT:    %3 = phi i32 [ %1, %[[CALL_0_LABEL]] ], [ %2, %[[CALL_1_LABEL]] ]
; CHECK-NEXT:    ret i32 %3

; THRESHOLD-0-LABEL:  define i32 @function_with_jump_table(i32 %index) {
; THRESHOLD-0:          %gep = getelementptr inbounds [2 x ptr], ptr @func_array, i32 0, i32 %index
; THRESHOLD-0-NEXT:     %func_ptr = load ptr, ptr %gep
; THRESHOLD-0-NEXT:     %result = call i32 %func_ptr()
; THRESHOLD-0-NEXT:     ret i32 %result
; THRESHOLD-0-NEXT:   }

define void @void_func0() {
  ret void
}

define void @void_func1() {
  ret void
}

@void_func_array = global [2 x ptr] [ptr @void_func0, ptr @void_func1]

define void @void_function_with_jump_table(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @void_func_array, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep
  call void %func_ptr()
  ret void
}

; CHECK-LABEL: define void @void_function_with_jump_table
; CHECK:       [[GEP:%.*]] = getelementptr inbounds [2 x ptr], ptr @void_func_array, i32 0, i32 %index
; CHECK:       switch i32 %index, label %[[VOID_DEFAULT_LABEL:.*]] [
; CHECK-NEXT:    i32 0, label %[[VOID_CALL_0_LABEL:.*]]
; CHECK-NEXT:    i32 1, label %[[VOID_CALL_1_LABEL:.*]]
; CHECK-NEXT:  ]

; CHECK:       [[VOID_DEFAULT_LABEL]]
; CHECK-NEXT:    unreachable

; CHECK:       [[VOID_CALL_0_LABEL]]:
; CHECK-NEXT:    call void @void_func0()
; CHECK-NEXT:    br label %[[VOID_TAIL_LABEL:.*]]

; CHECK:       [[VOID_CALL_1_LABEL]]:
; CHECK-NEXT:    call void @void_func1()
; CHECK-NEXT:    br label %[[VOID_TAIL_LABEL]]

; CHECK:       [[VOID_TAIL_LABEL]]:
; CHECK-NEXT:    ret void

; THRESHOLD-0-LABEL:  define void @void_function_with_jump_table(i32 %index) {
; THRESHOLD-0:          %gep = getelementptr inbounds [2 x ptr], ptr @void_func_array, i32 0, i32 %index
; THRESHOLD-0-NEXT:     %func_ptr = load ptr, ptr %gep
; THRESHOLD-0-NEXT:     call void %func_ptr()
; THRESHOLD-0-NEXT:     ret void
; THRESHOLD-0-NEXT:   }

