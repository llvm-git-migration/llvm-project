// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8 test-fastest-varying=0" -split-input-file | FileCheck %s

// CHECK-DAG: #[[$map_id0:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 12)>
// CHECK-DAG: #[[$map_id1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0 mod 16)>

// CHECK-LABEL: func @vec_affine_apply
func.func @vec_affine_apply(%arg0: memref<8x12x16xf32>, %arg1: memref<8x24x48xf32>) {
  affine.for %arg2 = 0 to 8 {
// CHECK: affine.for %[[S0:.*]] = 0 to 24 {
// CHECK-NEXT: affine.for %[[S1:.*]] = 0 to 48 step 8 {
    affine.for %arg3 = 0 to 24 {
      affine.for %arg4 = 0 to 48 {
// CHECK-NEXT: affine.apply #[[$map_id0]](%[[S0]])
// CHECK-NEXT: affine.apply #[[$map_id1]](%[[S1]])
        %0 = affine.apply affine_map<(d0) -> (d0 mod 12)>(%arg3)
        %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%arg4)
        %2 = affine.load %arg0[%arg2, %0, %1] : memref<8x12x16xf32>
        affine.store %2, %arg1[%arg2, %arg3, %arg4] : memref<8x24x48xf32>
      }
    }
  }
  return
}
