; RUN: opt < %s -passes=jump-table-to-switch -verify-dom-info -S | FileCheck %s

@func_array0 = global [2 x ptr] [ptr @func0, ptr @declared_only_func1]

define i32 @func0() {
  ret i32 1
}

declare i32 @declared_only_func1()

define i32 @function_with_jump_table0(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array0, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep, align 8
  %result = call i32 %func_ptr()
  ret i32 %result
}

; CHECK-LABEL:  define i32 @function_with_jump_table0(i32 %index) {
; CHECK:          %gep = getelementptr inbounds [2 x ptr], ptr @func_array0, i32 0, i32 %index
; CHECK-NEXT:     %func_ptr = load ptr, ptr %gep, align 8
; CHECK-NEXT:     %result = call i32 %func_ptr()
; CHECK-NEXT:     ret i32 %result
; CHECK-NEXT:   }

declare i32 @__gxx_personality_v0(...)

define i32 @function_with_jump_table1(i32 %index) personality ptr @__gxx_personality_v0 {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array0, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep, align 8
  %result = invoke i32 %func_ptr() to label %normal unwind label %exceptional
normal:
  ret i32 %result
exceptional:
  %landing_pad = landingpad { ptr, i32 } catch ptr null
  resume { ptr, i32 } %landing_pad
}

; CHECK-LABEL: define i32 @function_with_jump_table1(i32 %index) personality ptr @__gxx_personality_v0 {
; CHECK:         %gep = getelementptr inbounds [2 x ptr], ptr @func_array0, i32 0, i32 %index
; CHECK-NEXT:    %func_ptr = load ptr, ptr %gep, align 8
; CHECK-NEXT:    %result = invoke i32 %func_ptr()
; CHECK-NEXT:    to label %normal unwind label %exceptional
; CHECK:       normal:
; CHECK-NEXT:    ret i32 %result
; CHECK:       exceptional:
; CHECK-NEXT:    %landing_pad = landingpad { ptr, i32 }
; CHECK-NEXT:    catch ptr null
; CHECK:       resume { ptr, i32 } %landing_pad
; CHECK-NEXT: }

@func_array1 = global [1 x ptr] [ptr @func2]

define i32 @func2(i32 %arg) {
  ret i32 %arg
}

define i32 @function_with_jump_table2(i32 %index) {
  %gep = getelementptr inbounds [1 x ptr], ptr @func_array1, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep, align 8
  %result = musttail call i32 %func_ptr(i32 %index)
  ret i32 %result
}

; CHECK-LABEL: define i32 @function_with_jump_table2(i32 %index) {
; CHECK:         %gep = getelementptr inbounds [1 x ptr], ptr @func_array1, i32 0, i32 %index
; CHECK-NEXT:    %func_ptr = load ptr, ptr %gep, align 8
; CHECK-NEXT:    %result = musttail call i32 %func_ptr(i32 %index)
; CHECK-NEXT:    ret i32 %result
; CHECK-NEXT:  }

