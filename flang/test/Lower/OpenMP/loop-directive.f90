! This test checks lowering of OpenMP loop Directive.

! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: omp.private {type = private} @[[I_PRIV:.*]] : !fir.ref<i32>

! CHECK-LABEL: func.func @_QPtest_no_clauses
subroutine test_no_clauses()
  integer :: i, dummy = 1

  ! CHECK: omp.loop private(@[[I_PRIV]] %{{.*}}#0 -> %[[ARG:.*]] : !fir.ref<i32>) {
  ! CHECK:   omp.loop_nest (%[[IV:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) {{.*}} {
  ! CHECK:     %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG]]
  ! CHECK:     fir.store %[[IV]] to %[[ARG_DECL]]#1 : !fir.ref<i32>
  ! CHECK:   }
  ! CHECK: }
  !$omp loop
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine

