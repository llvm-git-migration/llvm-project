! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! UNSUPPORTED: system-windows

!$omp taskwait
end
