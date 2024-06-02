// RUN: mlir-opt %s --test-scf-vectorize=vector-bitwidth=128 -split-input-file | FileCheck %s

// CHECK-LABEL: @test
//  CHECK-SAME:  (%[[A:.*]]: memref<?xi32>, %[[B:.*]]: memref<?xi32>, %[[C:.*]]: memref<?xi32>) {
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?xi32>
//       CHECK:  %[[C4:.*]] = arith.constant 4 : index
//       CHECK:  %[[COUNT:.*]] = arith.ceildivsi %[[DIM]], %[[C4]] : index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%[[COUNT]]) step (%{{.*}}) {
//       CHECK:  %[[MULT:.*]] = arith.muli %[[I]], %[[C4]] : index
//       CHECK:  %[[M:.*]] = arith.subi %[[DIM]], %[[MULT]] : index
//       CHECK:  %[[MASK:.*]] = vector.create_mask %[[M]] : vector<4xi1>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xi32>
//       CHECK:  %[[A_VAL:.*]] = vector.maskedload %[[A]][%[[MULT]]], %[[MASK]], %[[P]] : memref<?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xi32>
//       CHECK:  %[[B_VAL:.*]] = vector.maskedload %[[B]][%[[MULT]]], %[[MASK]], %[[P]] : memref<?xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : vector<4xi32>
//       CHECK:  vector.maskedstore %[[C]][%1], %[[MASK]], %[[RES]] : memref<?xi32>, vector<4xi1>, vector<4xi32>
//       CHECK:  scf.reduce
func.func @test(%A: memref<?xi32>, %B: memref<?xi32>, %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = memref.dim %A, %c0 : memref<?xi32>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    %1 = memref.load %A[%i] : memref<?xi32>
    %2 = memref.load %B[%i] : memref<?xi32>
    %3 =  arith.addi %1, %2 : i32
    memref.store %3, %C[%i] : memref<?xi32>
  }
  return
}
