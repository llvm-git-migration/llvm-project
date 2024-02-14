! Test delayed privatization for the `private` clause.

! RUN: bbc -emit-fir -hlfir=false -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_firstprivate
  implicit none
  integer :: var1

!$OMP PARALLEL FIRSTPRIVATE(var1)
  var1 = 10
!$OMP END PARALLEL
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[VAR1_PRIVATIZER_SYM:.*]] : !fir.ref<i32> alloc {
! CHECK: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<i32>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<i32>):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]] : !fir.ref<i32>
! CHECK:    fir.store %[[ORIG_VAL]] to %[[PRIV_PRIV_ARG]] : !fir.ref<i32>
! CHECK:    omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<i32>)
! CHECK: }

! CHECK-LABEL: @_QPdelayed_privatization_firstprivate
! CHECK: omp.parallel private(@[[VAR1_PRIVATIZER_SYM]] %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
! CHECK: omp.terminator

