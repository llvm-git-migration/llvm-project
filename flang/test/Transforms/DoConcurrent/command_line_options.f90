! RUN: %flang --help | FileCheck %s --check-prefix=FLANG

! FLANG:      -fdo-concurrent-parallel=<value>
! FLANG-NEXT:   Try to map `do concurrent` loops to OpenMP [none|host|device] 

! RUN: bbc --help | FileCheck %s --check-prefix=BBC

! BBC:      -fdo-concurrent-parallel=<string>
! BBC-SAME:   Try to map `do concurrent` loops to OpenMP [none|host|device] 

! RUN: not %flang -fdo-concurrent-parallel=host %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=OPT

! OPT: error: lowering `do concurrent` loops to OpenMP is only supported if OpenMP is enabled.
! OPT-SAME: Enable OpenMP using `-fopenmp`.

program test_cli
end program
