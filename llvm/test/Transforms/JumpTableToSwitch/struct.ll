; RUN: opt < %s -passes=jump-table-to-switch -verify-dom-info -S | FileCheck %s

%"struct_ty" = type { [2 x ptr] }

@func_array = global %"struct_ty" { [2 x ptr] [ptr @func0, ptr @func1] }

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
