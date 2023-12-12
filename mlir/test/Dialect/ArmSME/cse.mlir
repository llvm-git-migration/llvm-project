// RUN: mlir-opt %s -cse -split-input-file | FileCheck %s

// These tests check that CSE does not remove 'arm_sme.zero/get_tile' ops as
// duplicates.

// CHECK-LABEL: @zero_tile
// CHECK: %[[TILE_0:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
// CHECK: %[[TILE_1:.*]] = arm_sme.zero : vector<[4]x[4]xi32>
func.func @zero_tile() {
  %tile_1 = arm_sme.zero : vector<[4]x[4]xi32>
  %tile_2 = arm_sme.zero : vector<[4]x[4]xi32>
  return
}

// -----

// CHECK-LABEL: @get_tile
// CHECK: %[[TILE_0:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
// CHECK: %[[TILE_1:.*]] = arm_sme.get_tile : vector<[4]x[4]xi32>
func.func @get_tile() {
  %tile_1 = arm_sme.get_tile : vector<[4]x[4]xi32>
  %tile_2 = arm_sme.get_tile : vector<[4]x[4]xi32>
  return
}

// -----

// Operation is pure and should be removed if it's trivially dead.

// CHECK-LABEL: @dead_outerproduct
// CHECK-NOT: arm_sme.outerproduct
func.func @dead_outerproduct(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>) {
  %0 = arm_sme.outerproduct %lhs, %rhs : vector<[4]xf32>, vector<[4]xf32>
  return
}
