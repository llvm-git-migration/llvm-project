      subroutine foo(a, b)
      if ( (a  .eq. 0) .and.(b. eq. 1)) then
         
         print *, "foo"
      end if
      end subroutine

! RUN: %flang_fc1 -fsyntax-only "-x" "f95-fixed" %s 2>&1 | FileCheck %s --allow-empty --check-prefix=F95-FIXED
! F95-FIXED-NOT: Could not parse {{.*}}x-f95-fixed.f
! F95-FIXED-NOT: error
! F95-FIXED-NOT: warning

! RUN: not %flang_fc1 -fsyntax-only "-x" "f95" %s 2>&1 | FileCheck %s --check-prefix=F95 --strict-whitespace
! F95: error: Could not parse {{.*}}x-f95-fixed.f
! F95: {{.*}}x-f95-fixed.f:2:31: error: expected ')'
! F95:       if ( (a  .eq. 0) .and.(b. eq. 1)) then
! F95: {{^([ ]{32})}}^
