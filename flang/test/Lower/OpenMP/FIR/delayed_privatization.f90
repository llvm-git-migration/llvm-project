subroutine private_clause_allocatable()
        integer :: xxx

!$OMP PARALLEL FIRSTPRIVATE(xxx)
    xxx = xxx + 2
!$OMP END PARALLEL

end subroutine

! This is what flang emits with the PoC:
! --------------------------------------
!
!func.func @_QPprivate_clause_allocatable() {
!  %0 = fir.alloca i32 {bindc_name = "xxx", uniq_name = "_QFprivate_clause_allocatableExxx"}
!  %1 = fir.alloca i32 {bindc_name = "yyy", uniq_name = "_QFprivate_clause_allocatableEyyy"}
!  omp.parallel {
!    %2 = fir.alloca i32 {bindc_name = "xxx", pinned, uniq_name = "_QFprivate_clause_allocatableExxx"}
!    %3 = fir.load %0 : !fir.ref<i32>
!    fir.store %3 to %2 : !fir.ref<i32>
!    %4 = fir.alloca i32 {bindc_name = "yyy", pinned, uniq_name = "_QFprivate_clause_allocatableEyyy"}
!    %5 = fir.load %1 : !fir.ref<i32>
!    fir.store %5 to %4 : !fir.ref<i32>
!    omp.terminator
!  }
!  return
!}
!
!"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "xxx.privatizer"}> ({
!^bb0(%arg0: !fir.ref<i32>):
!  %0 = fir.alloca i32 {bindc_name = "xxx", pinned, uniq_name = "_QFprivate_clause_allocatableExxx"}
!  %1 = fir.load %arg0 : !fir.ref<i32>
!  fir.store %1 to %0 : !fir.ref<i32>
!  omp.yield(%0 : !fir.ref<i32>)
!}) : () -> ()
!
!"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "yyy.privatizer"}> ({
!^bb0(%arg0: !fir.ref<i32>):
!  %0 = fir.alloca i32 {bindc_name = "yyy", pinned, uniq_name = "_QFprivate_clause_allocatableEyyy"}
!  %1 = fir.load %arg0 : !fir.ref<i32>
!  fir.store %1 to %0 : !fir.ref<i32>
!  omp.yield(%0 : !fir.ref<i32>)
!}) : () -> ()
