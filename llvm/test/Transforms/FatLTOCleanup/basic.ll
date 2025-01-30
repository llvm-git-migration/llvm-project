
; RUN: opt -passes="fatlto-cleanup" -mtriple=x86_64-unknown-fuchsia < %s -S | FileCheck %s



define hidden void @foo(ptr %p1) {
entry:
  %vtable = load ptr, ptr %p1, align 8
  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"_ZTS1a")
  %1 = extractvalue { ptr, i1 } %0, 1
  br i1 %1, label %cont2, label %trap1

trap1:
  tail call void @llvm.ubsantrap(i8 2)
  unreachable

cont2:
  %2 = extractvalue { ptr, i1 } %0, 0
  %call = tail call noundef i64 %2(ptr noundef nonnull align 8 dereferenceable(8) %p1)
  ret void
}

; CHECK-LABEL: define hidden void @foo
;  CHECK-NEXT: entry:
;  CHECK-NEXT:  %vtable = load ptr, ptr %p1, align 8
;  CHECK-NEXT:  %0 = getelementptr i8, ptr %vtable, i32 0
;  CHECK-NEXT:  %vfunc = load ptr, ptr %0, align 8
;  CHECK-NEXT:  br i1 true, label %cont2, label %trap1

; CHECK-LABEL: trap1:
;  CHECK-NEXT:  tail call void @llvm.ubsantrap(i8 2)
;  CHECK-NEXT:  unreachable

; CHECK-LABEL: cont2:
;  CHECK-NEXT:  %call = tail call noundef i64 %vfunc(ptr noundef nonnull align 8 dereferenceable(8) %p1)
;  CHECK-NEXT:  ret void
;  CHECK-NEXT:}

; Function Attrs: cold noreturn nounwind
declare void @llvm.ubsantrap(i8 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata) #1

attributes #0 = { cold noreturn nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
