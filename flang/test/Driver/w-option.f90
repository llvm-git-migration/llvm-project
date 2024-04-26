! RUN: %flang -c %s 2>&1 | FileCheck %s
! RUN: %flang -c -w %s 2>&1 | FileCheck %s -check-prefix=CHECK-W --allow-empty
! CHECK: warning: Label '40' is in a construct that should not be used as a branch target here
! CHECK: warning: Label '50' is in a construct that should not be used as a branch target here
! CHECK-W-NOT: warning

subroutine sub01(n)
  integer n
  GOTO (40,50,60) n
  if (n .eq. 1) then
40   print *, "xyz"
50 end if
60 continue
end subroutine sub01
