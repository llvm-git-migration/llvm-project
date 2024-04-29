! RUN: %flang_fc1 -pedantic %s 2>&1 | FileCheck %s

!CHECK: portability: deprecated usage
!CHECK: portability: deprecated usage
!CHECK: portability: passing Hollerith to unlimited polymorphic as if it were CHARACTER

module m
 contains
  subroutine unlimited(x)
    class(*), intent(in) :: x
  end
  subroutine test
    call unlimited(6HHERMAN)
    call unlimited('abc') ! ok
  end
end
