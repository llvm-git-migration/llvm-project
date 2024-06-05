! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in ALLOCATE statements

! C934
! If type-spec appears, it shall specify a type with which each
! allocate-object is type compatible.
! Issue #78939: allocatable object has a non-defined character length.

module m1
  integer::nn=1
end module m1

program main
  use m1
  character(nn),pointer::c1s
  !ERROR: Character length of allocatable object in ALLOCATE must be the same as the type-spec
  allocate(character(2)::c1s)
end program main
