!RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -DSTEP=1 %s
!RUN: %flang_fc1 -fsyntax-only -DSTEP=2 %s
!RUN: not %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s

! Tests that a module captured in a hermetic module file is compatible when
! USE'd with a module of the same name USE'd directly.

#if STEP == 1
module modfile71a
  ! not errors
  integer, parameter :: good_named_const = 123
  integer :: good_var = 1
  type :: good_derived
    integer component
  end type
  procedure(), pointer :: good_proc_ptr
  generic :: gen => bad_subroutine
  ! errors
  integer, parameter :: bad_named_const = 123
  integer :: bad_var = 1
  type :: bad_derived
    integer component
  end type
  procedure(), pointer :: bad_proc_ptr
 contains
  subroutine good_subroutine
  end
  subroutine bad_subroutine(x)
    integer x
  end
end

module modfile71b
  use modfile71a ! capture hermetically
end

#elif STEP == 2
module modfile71a
  ! not errors
  integer, parameter :: good_named_const = 123
  integer :: good_var = 1
  type :: good_derived
    integer component
  end type
  procedure(), pointer :: good_proc_ptr
  generic :: gen => bad_subroutine
  ! errors
  integer, parameter :: bad_named_const = 666
  real :: bad_var = 1.
  type :: bad_derived
    real component
  end type
  real, pointer :: bad_proc_ptr
 contains
  subroutine good_subroutine
  end
  subroutine bad_subroutine(x)
    real x
  end
end

#else

!CHECK: error: 'bad_derived' use-associated from 'bad_derived' in module 'modfile71a' is incompatible with 'bad_derived' from another module
!CHECK: error: 'bad_named_const' use-associated from 'bad_named_const' in module 'modfile71a' is incompatible with 'bad_named_const' from another module
!CHECK: error: 'bad_proc_ptr' use-associated from 'bad_proc_ptr' in module 'modfile71a' is incompatible with 'bad_proc_ptr' from another module
!CHECK: error: 'bad_subroutine' use-associated from 'bad_subroutine' in module 'modfile71a' is incompatible with 'bad_subroutine' from another module
!CHECK: error: 'bad_var' use-associated from 'bad_var' in module 'modfile71a' is incompatible with 'bad_var' from another module
!CHECK: warning: 'good_derived' is use-associated from 'good_derived' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_named_const' is use-associated from 'good_named_const' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_proc_ptr' is use-associated from 'good_proc_ptr' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_subroutine' is use-associated from 'good_subroutine' in two distinct instances of module 'modfile71a'
!CHECK: warning: 'good_var' is use-associated from 'good_var' in two distinct instances of module 'modfile71a'
!CHECK-NOT: error:
!CHECK-NOT: warning:

program main
  use modfile71a
  use modfile71b
end
#endif
