// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-linux-gnu -emit-llvm -o - | FileCheck  %s
// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-linux-gnu -emit-llvm -o - | FileCheck -check-prefix=CHECK-NO-MERGE-CONSTANTS %s
// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-linux-gnu -fmerge-all-constants -emit-llvm -o - | FileCheck -check-prefix=CHECK-MERGE-CONSTANTS %s

// CHECK-NO-MERGE-CONSTANTS: @{{.*}}.a1 = private unnamed_addr constant [5 x i32] [i32 0, i32 1, i32 2, i32 0, i32 0]

// CHECK-MERGE-CONSTANTS: @{{.*}}.a1 = internal constant [5 x i32] [i32 0, i32 1, i32 2, i32 0, i32 0]
// CHECK-MERGE-CONSTANTS: @{{.*}}.a2 = internal constant [5 x i32] zeroinitializer
// CHECK-MERGE-CONSTANTS: @{{.*}}.a3 = internal constant [5 x i32] zeroinitializer

void testConstArrayInits(void)
{
  const int a1[5] = {0,1,2};
  const int a2[5] = {0,0,0};
  const int a3[5] = {0};
}


// CHECK-LABEL: @testConstLongArrayInits()
// CHECK: entry:
// CHECK-NEXT:  %a1 = alloca [20 x i32], align 16
// CHECK-NEXT:  %a2 = alloca [20 x %struct.anon], align 16
// CHECK-NEXT:  %a3 = alloca [20 x %struct.anon.0], align 16
// CHECK-NEXT:  %a4 = alloca [20 x %struct.anon.1], align 16
//
// CHECK-NEXT:  %0 = getelementptr inbounds i8, ptr %a1, i64 8
// CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 8 %0, i8 0, i64 72, i1 false)
// CHECK-NEXT:  %1 = getelementptr inbounds <{ i32, i32, [18 x i32] }>, ptr %a1, i32 0, i32 0
// CHECK-NEXT:  store i32 1, ptr %1, align 16
// CHECK-NEXT:  %2 = getelementptr inbounds <{ i32, i32, [18 x i32] }>, ptr %a1, i32 0, i32 1
// CHECK-NEXT:  store i32 2, ptr %2, align 4
//
// CHECK-NEXT:  %3 = getelementptr inbounds i8, ptr %a2, i64 8
// CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 8 %3, i8 0, i64 152, i1 false)
// CHECK-NEXT:  %4 = getelementptr inbounds <{ %struct.anon, [19 x %struct.anon] }>, ptr %a2, i32 0, i32 0
// CHECK-NEXT:  %5 = getelementptr inbounds %struct.anon, ptr %4, i32 0, i32 0
// CHECK-NEXT:  store i8 1, ptr %5, align 16
// CHECK-NEXT:  %6 = getelementptr inbounds %struct.anon, ptr %4, i32 0, i32 1
// CHECK-NEXT:  store i32 2, ptr %6, align 4
//
// CHECK-NEXT:  %7 = getelementptr inbounds i8, ptr %a3, i64 1
// CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 1 %7, i8 0, i64 159, i1 false)
// CHECK-NEXT:  %8 = getelementptr inbounds <{ %struct.anon.0, [19 x %struct.anon.0] }>, ptr %a3, i32 0, i32 0
// CHECK-NEXT:  %9 = getelementptr inbounds %struct.anon.0, ptr %8, i32 0, i32 0
// CHECK-NEXT:  store i8 1, ptr %9, align 16
//
// CHECK-NEXT:  %10 = getelementptr inbounds i8, ptr %a4, i64 8
// CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 8 %10, i8 0, i64 392, i1 false)
// CHECK-NEXT:  %11 = getelementptr inbounds <{ %struct.anon.1, [19 x %struct.anon.1] }>, ptr %a4, i32 0, i32 0
// CHECK-NEXT:  %12 = getelementptr inbounds %struct.anon.1, ptr %11, i32 0, i32 0
// CHECK-NEXT:  store i8 1, ptr %12, align 16
// CHECK-NEXT:  %13 = getelementptr inbounds %struct.anon.1, ptr %11, i32 0, i32 1
// CHECK-NEXT:  %14 = getelementptr inbounds [4 x i32], ptr %13, i32 0, i32 0
// CHECK-NEXT:  store i32 2, ptr %14, align 4
//
// CHECK-NEXT:  ret void
// }

void testConstLongArrayInits(void)
{
   const int a1[20] = {1,2};
   const struct {char c; int i;} a2[20] = {{1,2}};
   const struct {char c; int i;} a3[20] = {{1}};
   const struct {char c; int i[4];} a4[20] = {{1,{2}}};
}
