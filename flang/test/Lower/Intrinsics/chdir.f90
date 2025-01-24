! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test_chdir()
  implicit none
! CHECK-LABEL:   func.func @_QPtest_chdir() {

  call chdir("..")
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QQclX2E2E) : !fir.ref<!fir.char<1,2>>
! CHECK:  %[[C_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_1:.*]] = hlfir.declare %[[VAL_0]] typeparams %[[C_2]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX2E2E"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:  %[[VAL_2:.*]] = fir.absent !fir.ref<none>
! CHECK:  %[[VAL_3:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,2>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %{{.*}} : (!fir.ref<none>) -> !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAChdir(%[[VAL_3]], %[[VAL_4]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i32>) -> none
end subroutine
