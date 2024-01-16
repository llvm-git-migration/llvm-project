
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsb
subroutine sb(arr)
  integer :: arr(:)
!CHECK: omp.parallel  {
  !$omp parallel
!CHECK: omp.single  {
  !$omp workshare
    arr = 0
  !$omp end workshare
!CHECK: }
  !$omp end parallel
!CHECK: }
end subroutine
