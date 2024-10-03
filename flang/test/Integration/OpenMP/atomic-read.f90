!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK: %[[VAL_1:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: %[[VAL_2:.*]] = alloca { float, float }, i64 1, align 8
!CHECK: %[[ATOMIC_TEMP_LOAD:.*]] = alloca { float, float }, align 16
!CHECK: call void @__atomic_load(i64 8, ptr %[[VAL_2]], ptr %[[ATOMIC_TEMP_LOAD]], i32 0)
!CHECK: %[[VAL_3:.*]] = load { float, float }, ptr %[[ATOMIC_TEMP_LOAD]], align 16
!CHECK: store { float, float } %[[VAL_3]], ptr %[[VAL_1]], align 4

program main
      complex*8 ia, ib
      !$omp atomic read
        ib = ia
end program
