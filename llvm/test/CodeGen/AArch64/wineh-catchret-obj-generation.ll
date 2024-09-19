; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype=obj %s -o %t.o

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.seh.try.begin() #0

define fastcc ptr @test_function(i1 %0, ptr %_Fmtfl.i.i, i1 %1) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %0, label %right-block527, label %left-block526

common.ret1:
  %common.ret1.op = phi ptr [ null, %left-block530 ], [ null, %some-block ], [ %_Fmtfl.i.i, %invoke.cont.i124 ], [ null, %left-block526 ]
  ret ptr %common.ret1.op

invoke.cont.i124:
  %.not657 = icmp eq i32 1, 0
  br i1 %.not657, label %some-block, label %common.ret1

catch.dispatch.i:
  %2 = catchswitch within none [label %catch.i] unwind to caller

catch.i:
  %3 = catchpad within %2 [ptr null, i32 0, ptr null]
  catchret from %3 to label %some-block

some-block:
  br label %common.ret1

left-block526:
  br i1 %1, label %common.ret1, label %left-block530

right-block527:
  invoke void @llvm.seh.try.begin()
          to label %invoke.cont.i124 unwind label %catch.dispatch.i

left-block530:
  %.not = icmp eq i32 0, 0
  br label %common.ret1
}

attributes #0 = { nounwind willreturn memory(write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"eh-asynch", i32 1}
