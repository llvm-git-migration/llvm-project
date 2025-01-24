! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: func @_QPtest_ierrno(
subroutine test_ierrno(name)
    integer :: i
    i = ierrno()
! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_ierrnoEi"}
! CHECK: %[[VAL_1:.*]] = fir.call @_FortranAIerrno
! CHECK: fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK: return
end subroutine test_ierrno
