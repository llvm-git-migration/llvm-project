! REQUIRES: amdgpu-registered-target

! Test that AMDGPU-specific flang-new OpenMP offload related commands expand to
! the appropriate commands for flang-new -fc1 as expected. Contrary to tests
! located in omp-driver-offload.f90, driver tests here do require the amdgcn-
! amd-amdhsa triple to be recognized.

! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: --target=x86_64-unknown-linux-gnu \
! RUN: | FileCheck %s --check-prefix=OFFLOAD-TARGETS

! OFFLOAD-TARGETS: "{{[^"]*}}flang-new" "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! OFFLOAD-TARGETS-SAME: "-fopenmp-targets=amdgcn-amd-amdhsa"
! OFFLOAD-TARGETS-NEXT: "{{[^"]*}}flang-new" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OFFLOAD-TARGETS-NOT: -fopenmp-targets
! OFFLOAD-TARGETS: "{{[^"]*}}flang-new" "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! OFFLOAD-TARGETS-SAME: "-fopenmp-targets=amdgcn-amd-amdhsa"
