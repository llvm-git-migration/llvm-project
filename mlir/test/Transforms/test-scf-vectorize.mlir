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

// -----

// CHECK-LABEL: @test
//  CHECK-SAME:  (%[[A:.*]]: memref<?xindex>, %[[B:.*]]: memref<?xindex>, %[[C:.*]]: memref<?xindex>) {
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?xindex>
//       CHECK:  %[[C4:.*]] = arith.constant 4 : index
//       CHECK:  %[[COUNT:.*]] = arith.ceildivsi %[[DIM]], %[[C4]] : index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%[[COUNT]]) step (%{{.*}}) {
//       CHECK:  %[[MULT:.*]] = arith.muli %[[I]], %[[C4]] : index
//       CHECK:  %[[M:.*]] = arith.subi %[[DIM]], %[[MULT]] : index
//       CHECK:  %[[MASK:.*]] = vector.create_mask %[[M]] : vector<4xi1>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xindex>
//       CHECK:  %[[A_VAL:.*]] = vector.maskedload %[[A]][%[[MULT]]], %[[MASK]], %[[P]] : memref<?xindex>, vector<4xi1>, vector<4xindex> into vector<4xindex>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xindex>
//       CHECK:  %[[B_VAL:.*]] = vector.maskedload %[[B]][%[[MULT]]], %[[MASK]], %[[P]] : memref<?xindex>, vector<4xi1>, vector<4xindex> into vector<4xindex>
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : vector<4xindex>
//       CHECK:  vector.maskedstore %[[C]][%1], %[[MASK]], %[[RES]] : memref<?xindex>, vector<4xi1>, vector<4xindex>
//       CHECK:  scf.reduce

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {
func.func @test(%A: memref<?xindex>, %B: memref<?xindex>, %C: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = memref.dim %A, %c0 : memref<?xindex>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    %1 = memref.load %A[%i] : memref<?xindex>
    %2 = memref.load %B[%i] : memref<?xindex>
    %3 =  arith.addi %1, %2 : index
    memref.store %3, %C[%i] : memref<?xindex>
  }
  return
}
}

// -----

func.func private @non_vectorizable(i32) -> (i32)

// CHECK-LABEL: @test
//  CHECK-SAME:  (%[[A:.*]]: memref<?xi32>, %[[B:.*]]: memref<?xi32>, %[[C:.*]]: memref<?xi32>) {
//       CHECK:  %[[C00:.*]] = arith.constant 0 : index
//       CHECK:  %[[DIM:.*]] = memref.dim %[[A]], %[[C00]] : memref<?xi32>
//       CHECK:  %[[C4:.*]] = arith.constant 4 : index
//       CHECK:  %[[COUNT:.*]] = arith.divsi %[[DIM]], %[[C4]] : index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%[[COUNT]]) step (%{{.*}}) {
//       CHECK:  %[[MULT:.*]] = arith.muli %[[I]], %[[C4]] : index
//       CHECK:    %[[A_VAL:.*]] = vector.load %[[A]][%[[MULT]]] : memref<?xi32>, vector<4xi32>
//       CHECK:    %[[B_VAL:.*]] = vector.load %[[B]][%[[MULT]]] : memref<?xi32>, vector<4xi32>
//       CHECK:    %[[R1:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : vector<4xi32>
//       CHECK:    %[[C0:.*]] = arith.constant 0 : index
//       CHECK:    %[[E0:.*]] = vector.extractelement %[[R1]][%[[C0]] : index] : vector<4xi32>
//       CHECK:    %[[C1:.*]] = arith.constant 1 : index
//       CHECK:    %[[E1:.*]] = vector.extractelement %[[R1]][%[[C1]] : index] : vector<4xi32>
//       CHECK:    %[[C2:.*]] = arith.constant 2 : index
//       CHECK:    %[[E2:.*]] = vector.extractelement %[[R1]][%[[C2]] : index] : vector<4xi32>
//       CHECK:    %[[C3:.*]] = arith.constant 3 : index
//       CHECK:    %[[E3:.*]] = vector.extractelement %[[R1]][%[[C3]] : index] : vector<4xi32>
//       CHECK:    %[[R2:.*]] = func.call @non_vectorizable(%[[E0]]) : (i32) -> i32
//       CHECK:    %[[R3:.*]] = func.call @non_vectorizable(%[[E1]]) : (i32) -> i32
//       CHECK:    %[[R4:.*]] = func.call @non_vectorizable(%[[E2]]) : (i32) -> i32
//       CHECK:    %[[R5:.*]] = func.call @non_vectorizable(%[[E3]]) : (i32) -> i32
//       CHECK:    %[[RES1:.*]] = ub.poison : vector<4xi32>
//       CHECK:    %[[C0:.*]] = arith.constant 0 : index
//       CHECK:    %[[RES2:.*]] = vector.insertelement %[[R2]], %[[RES1]][%[[C0]] : index] : vector<4xi32>
//       CHECK:    %[[C1:.*]] = arith.constant 1 : index
//       CHECK:    %[[RES3:.*]] = vector.insertelement %[[R3]], %[[RES2]][%[[C1]] : index] : vector<4xi32>
//       CHECK:    %[[C2:.*]] = arith.constant 2 : index
//       CHECK:    %[[RES4:.*]] = vector.insertelement %[[R4]], %[[RES3]][%[[C2]] : index] : vector<4xi32>
//       CHECK:    %[[C3:.*]] = arith.constant 3 : index
//       CHECK:    %[[RES5:.*]] = vector.insertelement %[[R5]], %[[RES4]][%[[C3]] : index] : vector<4xi32>
//       CHECK:    vector.store %[[RES5]], %[[C]][%[[MULT]]] : memref<?xi32>, vector<4xi32>
//       CHECK:    scf.reduce
//       CHECK:  }
//       CHECK:  %[[UB1:.*]] = arith.muli %[[COUNT]], %[[C4]] : index
//       CHECK:  %[[UB2:.*]] = arith.addi %[[UB1]], %[[C00]] : index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%[[UB2]]) to (%[[DIM]]) step (%{{.*}}) {
//       CHECK:    %[[A_VAL:.*]] = memref.load %[[A]][%[[I]]] : memref<?xi32>
//       CHECK:    %[[B_VAL:.*]] = memref.load %[[B]][%[[I]]] : memref<?xi32>
//       CHECK:    %[[R1:.*]] = arith.addi %[[A_VAL:.*]], %[[B_VAL:.*]] : i32
//       CHECK:    %[[R2:.*]] = func.call @non_vectorizable(%[[R1]]) : (i32) -> i32
//       CHECK:    memref.store %[[R2]], %[[C]][%[[I]]] : memref<?xi32>
//       CHECK:    scf.reduce
//       CHECK:  }
func.func @test(%A: memref<?xi32>, %B: memref<?xi32>, %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = memref.dim %A, %c0 : memref<?xi32>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    %1 = memref.load %A[%i] : memref<?xi32>
    %2 = memref.load %B[%i] : memref<?xi32>
    %3 =  arith.addi %1, %2 : i32
    %4 = func.call @non_vectorizable(%3) : (i32) -> (i32)
    memref.store %4, %C[%i] : memref<?xi32>
  }
  return
}

// -----

// CHECK-LABEL: @test
//  CHECK-SAME:  (%[[A:.*]]: memref<?xindex>, %[[B:.*]]: memref<?xindex>, %[[C:.*]]: memref<?xindex>) {
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[C2:.*]] = arith.constant 2 : index
//       CHECK:  %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?xindex>
//       CHECK:  %[[C4:.*]] = arith.constant 4 : index
//       CHECK:  %[[COUNT:.*]] = arith.ceildivsi %[[DIM]], %[[C4]] : index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%[[COUNT]]) step (%{{.*}}) {
//       CHECK:  %[[OFFSETS:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//       CHECK:  %[[MULT:.*]] = arith.muli %[[I]], %[[C4]] : index
//       CHECK:  %[[O1:.*]] = vector.splat %[[MULT]] : vector<4xindex>
//       CHECK:  %[[O2:.*]] = arith.addi %[[O1]], %[[OFFSETS]] : vector<4xindex>
//       CHECK:  %[[O3:.*]] = vector.splat %[[C2]] : vector<4xindex>
//       CHECK:  %[[O4:.*]] = arith.muli %[[O2]], %[[O3]] : vector<4xindex>
//       CHECK:  %[[M:.*]] = arith.subi %[[DIM]], %[[MULT]] : index
//       CHECK:  %[[MASK:.*]] = vector.create_mask %[[M]] : vector<4xi1>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xindex>
//       CHECK:  %[[A_VAL:.*]] = vector.gather %arg0[%{{.*}}] [%[[O4]]], %[[MASK]], %[[P]] : memref<?xindex>, vector<4xindex>, vector<4xi1>, vector<4xindex> into vector<4xindex>
//       CHECK:  %[[P:.*]] = ub.poison : vector<4xindex>
//       CHECK:  %[[B_VAL:.*]] = vector.maskedload %[[B]][%[[MULT]]], %[[MASK]], %[[P]] : memref<?xindex>, vector<4xi1>, vector<4xindex> into vector<4xindex>
//       CHECK:  %[[RES:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : vector<4xindex>
//       CHECK:  vector.maskedstore %[[C]][%1], %[[MASK]], %[[RES]] : memref<?xindex>, vector<4xi1>, vector<4xindex>
//       CHECK:  scf.reduce

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {
func.func @test(%A: memref<?xindex>, %B: memref<?xindex>, %C: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %count = memref.dim %A, %c0 : memref<?xindex>
  scf.parallel (%i) = (%c0) to (%count) step (%c1) {
    %0 = arith.muli %i, %c2 : index
    %1 = memref.load %A[%0] : memref<?xindex>
    %2 = memref.load %B[%i] : memref<?xindex>
    %3 =  arith.addi %1, %2 : index
    memref.store %3, %C[%i] : memref<?xindex>
  }
  return
}
}
