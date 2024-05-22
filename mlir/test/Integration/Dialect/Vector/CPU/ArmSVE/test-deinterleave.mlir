// RUN: mlir-opt %s -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN: -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @entry() {
  %step_vector = llvm.intr.experimental.stepvector : vector<[4]xi8>
  vector.print %step_vector : vector<[4]xi8>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  %v1, %v2 = vector.deinterleave %step_vector : vector<[4]xi8> -> vector<[2]xi8>
  vector.print %v1 : vector<[2]xi8>
  vector.print %v2 : vector<[2]xi8>
  // CHECK: ( 0, 2, 4, 6 )
  // CHECK: ( 1, 3, 5, 7 )

  return
}
