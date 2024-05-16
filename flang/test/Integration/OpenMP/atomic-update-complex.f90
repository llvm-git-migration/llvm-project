!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK: define void @_QQmain() {
!CHECK: %[[VAL_1:.*]] = alloca { float, float }, align 8
!CHECK: %[[VAL_2:.*]] = alloca { float, float }, align 8
!CHECK: %[[VAL_3:.*]] = alloca { float, float }, align 8
!CHECK: %[[X_NEW_VAL:.*]] = alloca { float, float }, align 8
!CHECK: %[[VAL_4:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: %[[VAL_5:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: store { float, float } { float 2.000000e+00, float 2.000000e+00 }, ptr %[[VAL_4]], align 4
!CHECK: br label %entry

program main
      complex*8 ia, ib
      ia = (2, 2)

!CHECK: entry:
!CHECK: call void @llvm.lifetime.start.p0(i64 8, ptr %[[VAL_3]])
!CHECK: call void @__atomic_load(i64 8, ptr %[[VAL_4]], ptr %[[VAL_3]], i32 0)
!CHECK: %[[VAL_6:.*]] = load { float, float }, ptr %[[VAL_3]], align 8
!CHECK: call void @llvm.lifetime.end.p0(i64 8, ptr %[[VAL_3]])
!CHECK: br label %.atomic.cont


!CHECK: .atomic.cont
!CHECK: %[[VAL_7:.*]] = phi { float, float } [ %[[VAL_6]], %entry ], [ {{.*}}, %.atomic.cont ]
!CHECK: %[[VAL_8:.*]] = extractvalue { float, float } %[[VAL_7]], 0
!CHECK: %[[VAL_9:.*]] = extractvalue { float, float } %[[VAL_7]], 1
!CHECK: %[[VAL_10:.*]] = fadd contract float %[[VAL_8]], 1.000000e+00
!CHECK: %[[VAL_11:.*]] = fadd contract float %[[VAL_9]], 1.000000e+00
!CHECK: %[[VAL_12:.*]] = insertvalue { float, float } undef, float %[[VAL_10]], 0
!CHECK: %[[VAL_13:.*]] = insertvalue { float, float } %[[VAL_12]], float %[[VAL_11]], 1
!CHECK: store { float, float } %[[VAL_13]], ptr %[[X_NEW_VAL]], align 4
!CHECK: %[[VAL_14:.*]] = load { float, float }, ptr %[[X_NEW_VAL]], align 4
!CHECK: call void @llvm.lifetime.start.p0(i64 8, ptr %[[VAL_1]])
!CHECK: store { float, float } %[[VAL_7]], ptr %[[VAL_1]], align 8
!CHECK: call void @llvm.lifetime.start.p0(i64 8, ptr %[[VAL_2]])
!CHECK: store { float, float } %[[VAL_14]], ptr %[[VAL_2]], align 8
!CHECK: %[[VAL_15:.*]] = call zeroext i1 @__atomic_compare_exchange(i64 8, ptr %[[VAL_4]], ptr %[[VAL_1]], ptr %[[VAL_2]], i32 0, i32 0)
!CHECK: call void @llvm.lifetime.end.p0(i64 8, ptr %[[VAL_2]])
!CHECK: %[[VAL_16:.*]] = load { float, float }, ptr %[[VAL_1]], align 8
!CHECK: %[[VAL_17:.*]] = insertvalue { { float, float }, i1 } poison, { float, float } %[[VAL_16]], 0
!CHECK: %[[VAL_18:.*]] = insertvalue { { float, float }, i1 } %[[VAL_17]], i1 %[[VAL_15]], 1
!CHECK: %[[VAL_19:.*]] = extractvalue { { float, float }, i1 } %[[VAL_18]], 0
!CHECK: %[[VAL_20:.*]] = extractvalue { { float, float }, i1 } %[[VAL_18]], 1
!CHECK: br i1 %[[VAL_20]], label %.atomic.exit, label %.atomic.cont
      !$omp atomic update
        ia = ia + (1, 1)
      !$omp end atomic  
      print *, ia
end program main
