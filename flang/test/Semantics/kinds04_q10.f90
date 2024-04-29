! RUN: not %flang_fc1 -triple x86_64-unknown-linux-gnu -pedantic %s 2>&1 | FileCheck %s
! C716 If both kind-param and exponent-letter appear, exponent-letter
! shall be E. (As an extension we also allow an exponent-letter which matches
! the kind-param)
! C717 The value of kind-param shall specify an approximation method that
! exists on the processor.
!
! This test is for x86_64, where exponent-letter 'q' is for
! 10-byte extended precision
! UNSUPPORTED: system-windows, system-aix
! REQUIRES: x86-registered-target

!CHECK: portability: nonstandard usage: Q exponent
!CHECK: portability: nonstandard usage: Q exponent
!CHECK: portability: nonstandard usage: Q exponent
!CHECK: portability: nonstandard usage: Q exponent
!CHECK: portability: nonstandard usage: Q exponent
!CHECK: portability: Explicit kind parameter together with non-'E' exponent letter is not standard
!CHECK: portability: Explicit kind parameter together with non-'E' exponent letter is not standard
!CHECK: warning: Explicit kind parameter on real constant disagrees with exponent letter 'q'
!CHECK: error: Unsupported REAL(KIND=32)
!CHECK: portability: Explicit kind parameter together with non-'E' exponent letter is not standard
!CHECK: warning: Explicit kind parameter on real constant disagrees with exponent letter 'q'
!CHECK: error: Unsupported REAL(KIND=32)

subroutine s(var)
  real :: realvar1 = 4.0E6_4
  real :: realvar2 = 4.0D6
  real :: realvar3 = 4.0Q6
  real :: realvar4 = 4.0D6_8
  real :: realvar5 = 4.0Q6_10
  real :: realvar6 = 4.0Q6_16
  real :: realvar7 = 4.0E6_8
  real :: realvar8 = 4.0E6_10
  real :: realvar9 = 4.0E6_16
  real :: realvar10 = 4.0E6_32
  double precision :: doublevar1 = 4.0E6_4
  double precision :: doublevar2 = 4.0D6
  double precision :: doublevar3 = 4.0Q6
  double precision :: doublevar4 = 4.0D6_8
  double precision :: doublevar5 = 4.0Q6_16
  double precision :: doublevar6 = 4.0E6_8
  double precision :: doublevar7 = 4.0E6_10
  double precision :: doublevar8 = 4.0E6_16
  double precision :: doublevar9 = 4.0E6_32
end subroutine s
