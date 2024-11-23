!RUN: %flang -save-temps -### %S/Inputs/free-form-test.f90  2>&1 | FileCheck %s --check-prefix=FREE
!RUN: %flang -save-temps -### %S/Inputs/fixed-form-test.f  2>&1 | FileCheck %s --check-prefix=FIXED

FREE:       "-fc1" {{.*}} "-o" "free-form-test.i" {{.*}} "-x" "f95-cpp-input" "{{.*}}/free-form-test.f90"
FREE-NEXT:  "-fc1" {{.*}} "-o" "free-form-test.bc" {{.*}} "-x" "f95" "free-form-test.i"

FIXED:      "-fc1" {{.*}} "-o" "fixed-form-test.i" {{.*}} "-x" "f95-fixed-cpp-input" "{{.*}}/fixed-form-test.f"
FIXED-NEXT: "-fc1" {{.*}} "-o" "fixed-form-test.bc" {{.*}} "-x" "f95-fixed" "fixed-form-test.i"
