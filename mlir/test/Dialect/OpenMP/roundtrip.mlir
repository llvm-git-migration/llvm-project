// RUN: fir-opt -verify-diagnostics %s | fir-opt | FileCheck %s

// CHECK: omp.private @x.privatizer : (!fir.ref<i32>) -> !fir.ref<i32> {
omp.private @x.privatizer : (!fir.ref<i32>) -> !fir.ref<i32> {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !fir.ref<i32>):

  // CHECK: %0 = fir.alloca i32
  %0 = fir.alloca i32
  // CHECK: omp.yield(%0 : !fir.ref<i32>)
  omp.yield(%0 : !fir.ref<i32>)
}

