// RUN: fir-opt -verify-diagnostics %s | fir-opt | FileCheck %s

// CHECK: omp.private {type = private} @x.privatizer : !fir.ref<i32>(alloc {
omp.private {type = private} @x.privatizer : !fir.ref<i32> (alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !fir.ref<i32>):
  omp.yield(%arg0 : !fir.ref<i32>)
})

// CHECK: omp.private {type = firstprivate} @y.privatizer : !fir.ref<i32>(alloc {
omp.private {type = firstprivate} @y.privatizer : !fir.ref<i32> (alloc {
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !fir.ref<i32>):
  omp.yield(%arg0 : !fir.ref<i32>)
// CHECK: } copy {
} copy {
// CHECK: ^bb0(%arg0: {{.*}}, %arg1: {{.*}}):
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
  omp.yield(%arg0 : !fir.ref<i32>)
})

