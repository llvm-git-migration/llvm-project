! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
! CHECK: print *, 3.14159
      program main
#include "inc-contin-2.h"
     &14159
      end program main
