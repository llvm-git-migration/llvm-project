! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! OpenMP Version 4.5
! 2.8.1 simd Construct
! An ordered construct with the simd clause is the only OpenMP construct
! that can be encountered during execution of a simd region.

program omp_simd
  integer i, j, k
  integer, allocatable :: a(:)

  allocate(a(10))

  !$omp simd
  do i = 1, 10
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
    !$omp single
    a(i) = i
    !$omp end single
  end do
  !$omp end simd

  print *, a

end program omp_simd
