subroutine delayed_privatization()
  integer :: var1
  integer :: var2

!$OMP PARALLEL FIRSTPRIVATE(var1, var2)
  var1 = var1 + var2 + 2
!$OMP END PARALLEL

end subroutine

! This is what flang emits with the PoC:
! --------------------------------------
!
!func.func @_QPdelayed_privatization() {
!  %0 = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatizationEvar1"}
!  %1 = fir.alloca i32 {bindc_name = "var2", uniq_name = "_QFdelayed_privatizationEvar2"}
!  omp.parallel private(@var1.privatizer %0, @var2.privatizer %1 : !fir.ref<i32>, !fir.ref<i32>) {
!    %2 = fir.load %0 : !fir.ref<i32>
!    %3 = fir.load %1 : !fir.ref<i32>
!    %4 = arith.addi %2, %3 : i32
!    %c2_i32 = arith.constant 2 : i32
!    %5 = arith.addi %4, %c2_i32 : i32
!    fir.store %5 to %0 : !fir.ref<i32>
!    omp.terminator
!  }
!  return
!}
!
!"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var1.privatizer"}> ({
!^bb0(%arg0: !fir.ref<i32>):
!  %0 = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatizationEvar1"}
!  %1 = fir.load %arg0 : !fir.ref<i32>
!  fir.store %1 to %0 : !fir.ref<i32>
!  omp.yield(%0 : !fir.ref<i32>)
!}) : () -> ()
!
!"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var2.privatizer"}> ({
!^bb0(%arg0: !fir.ref<i32>):
!  %0 = fir.alloca i32 {bindc_name = "var2", pinned, uniq_name = "_QFdelayed_privatizationEvar2"}
!  %1 = fir.load %arg0 : !fir.ref<i32>
!  fir.store %1 to %0 : !fir.ref<i32>
!  omp.yield(%0 : !fir.ref<i32>)
!}) : () -> ()
