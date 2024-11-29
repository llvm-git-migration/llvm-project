!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

subroutine f00
  real :: x
!ERROR: The type of 'x' is incompatible with the reduction operator.
!$omp target in_reduction(.or.: x)
!$omp end target
end

subroutine f01
  real :: x
!ERROR: Invalid reduction operator in IN_REDUCTION clause.
!$omp target in_reduction(.not.: x)
!$omp end target
end

subroutine f02(p)
  integer, pointer, intent(in) :: p
!ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a IN_REDUCTION clause
!$omp target in_reduction(+: p)
!$omp end target
end

subroutine f03
  common /c/ a, b 
!ERROR: Common block names are not allowed in IN_REDUCTION clause
!$omp target in_reduction(+: /c/)
!$omp end target
end

subroutine f04
  integer :: x(10)
!ERROR: Reference to x must be a contiguous object
!$omp target in_reduction(+: x(1:10:2))
!$omp end target
end

subroutine f05
  integer :: x(10)
!ERROR: 'x' in IN_REDUCTION clause is a zero size array section
!$omp target in_reduction(+: x(1:0))
!$omp end target
end

