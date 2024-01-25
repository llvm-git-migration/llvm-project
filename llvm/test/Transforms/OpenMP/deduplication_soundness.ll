; RUN: opt -passes=openmp-opt-cgscc -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

declare i32 @printf(ptr noundef, ...)
declare i32 @omp_get_thread_limit()
; Function Attrs: nounwind
declare void @__kmpc_set_thread_limit(ptr, i32, i32)
; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr)
; Function Attrs: nounwind
declare noalias ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr)
; Function Attrs: nounwind
declare void @__kmpc_omp_task_complete_if0(ptr, i32, ptr)
; Function Attrs: nounwind
declare void @__kmpc_omp_task_begin_if0(ptr, i32, ptr)

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@.str = private unnamed_addr constant [28 x i8] c"\0A1:target thread_limit: %d\0A\00", align 1
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@.str.1 = private unnamed_addr constant [28 x i8] c"\0A2:target thread_limit: %d\0A\00", align 1

define dso_local i32 @main() local_unnamed_addr {
; CHECK-LABEL: define {{[^@]+}}@main
; CHECK-NEXT:  entry:
; CHECK: %call.i.i.i = call i32 @omp_get_thread_limit()
; CHECK-NEXT: %call1.i.i.i = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %call.i.i.i)
; CHECK: %call.i.i.i1 = call i32 @omp_get_thread_limit()
; CHECK-NEXT: %call1.i.i.i2 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %call.i.i.i1)
entry:
  %0 = call i32 @__kmpc_global_thread_num(ptr nonnull @1)
  %1 = call ptr @__kmpc_omp_task_alloc(ptr nonnull @1, i32 %0, i32 1, i64 40, i64 0, ptr nonnull @.omp_task_entry.)
  call void @__kmpc_omp_task_begin_if0(ptr nonnull @1, i32 %0, ptr %1)
  call void @__kmpc_set_thread_limit(ptr nonnull @1, i32 %0, i32 4)
  %call.i.i.i = call i32 @omp_get_thread_limit()
  %call1.i.i.i = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %call.i.i.i)
  call void @__kmpc_omp_task_complete_if0(ptr nonnull @1, i32 %0, ptr %1)
  %2 = call ptr @__kmpc_omp_task_alloc(ptr nonnull @1, i32 %0, i32 1, i64 40, i64 0, ptr nonnull @.omp_task_entry..3)
  call void @__kmpc_omp_task_begin_if0(ptr nonnull @1, i32 %0, ptr %2)
  call void @__kmpc_set_thread_limit(ptr nonnull @1, i32 %0, i32 3)
  %call.i.i.i1 = call i32 @omp_get_thread_limit()
  %call1.i.i.i2 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %call.i.i.i1)
  call void @__kmpc_omp_task_complete_if0(ptr nonnull @1, i32 %0, ptr %2)
  ret i32 0
}

define internal noundef i32 @.omp_task_entry.(i32 noundef %0, ptr noalias nocapture noundef readonly %1) {
entry:
  tail call void @__kmpc_set_thread_limit(ptr nonnull @1, i32 %0, i32 4)
  %call.i.i = tail call i32 @omp_get_thread_limit()
  %call1.i.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %call.i.i)
  ret i32 0
}

define internal noundef i32 @.omp_task_entry..3(i32 noundef %0, ptr noalias nocapture noundef readonly %1) {
entry:
  tail call void @__kmpc_set_thread_limit(ptr nonnull @1, i32 %0, i32 3)
  %call.i.i = tail call i32 @omp_get_thread_limit()
  %call1.i.i = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %call.i.i)
  ret i32 0
}

attributes #1 = { alwaysinline norecurse nounwind uwtable }
attributes #3 = { alwaysinline nounwind uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"openmp", i32 51}
