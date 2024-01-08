! RUN: %flang -### -pthread /dev/null -o /dev/null 2>&1 | FileCheck %s
! RUN: %flang -### -Xflang -pthread /dev/null -o /dev/null 2>&1 | FileCheck %s

! How the -pthread flag is handled is very platform-specific. A lot of that
! functionality is tested by clang, and the flag itself is handled by clang's
! driver that flang also uses. Instead of duplicating all that testing here,
! just check that the presence of the flag does not raise an error. If we need
! more customized handling of -pthread, the tests for that can be added here.
!
! CHECK-NOT: error:
