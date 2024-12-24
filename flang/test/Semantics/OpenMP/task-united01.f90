! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
!
! OpenMP 5.2: 5.2 threadprivate directive restriction

subroutine task_united01()
    integer, save :: var_01, var_02(2)
    real          :: var_03
    common /c/ var_03

    !$omp threadprivate(var_01, var_02)
    !$omp threadprivate(/c/)

    ! ERROR: A THREADPRIVATE variable `var_01` cannot appear in a UNTIED TASK region
    ! ERROR: A THREADPRIVATE variable `var_02` cannot appear in a UNTIED TASK region
    ! ERROR: A THREADPRIVATE variable `var_01` cannot appear in a UNTIED TASK region
    !$omp task untied
        var_01    = 10
        var_02(1) = sum([var_01, 20])
    !$omp end task

    ! ERROR: A THREADPRIVATE variable `var_02` cannot appear in a UNTIED TASK region
    ! ERROR: A THREADPRIVATE variable `var_03` cannot appear in a UNTIED TASK region
    !$omp task untied
        var_02(2) = product(var_02)
        var_03    = 3.14
    !$omp end task
end subroutine task_united01
