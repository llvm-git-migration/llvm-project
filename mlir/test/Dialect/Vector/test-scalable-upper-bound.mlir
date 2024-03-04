// RUN: mlir-opt %s -test-affine-reify-value-bounds -cse -verify-diagnostics \
// RUN:   -verify-diagnostics -split-input-file | FileCheck %s

#fixedDim0Map = affine_map<(d0)[s0] -> (-d0 + 32400, s0)>
#fixedDim1Map = affine_map<(d0)[s0] -> (-d0 + 16, s0)>

// Here the upper bound for min_i is 4 x vscale, as we know 4 x vscale is
// always less than 32400. The bound for min_j is 16 as at vscale > 4,
// 4 x vscale will be > 16, so the value will be clamped at 16.

// CHECK: #[[$SCALABLE_BOUND_MAP_0:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @fixed_size_loop_nest
//   CHECK-DAG:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_0]]()[%vscale]
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]], %[[C16]]) : (index, index) -> ()
func.func @fixed_size_loop_nest() {
  %c16 = arith.constant 16 : index
  %c32400 = arith.constant 32400 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c32400 step %c4_vscale {
    %min_i = affine.min #fixedDim0Map(%i)[%c4_vscale]
    scf.for %j = %c0 to %c16 step %c4_vscale {
      %min_j = affine.min #fixedDim1Map(%j)[%c4_vscale]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB"} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB"} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

#dynamicDim0Map = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>
#dynamicDim1Map = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>

// Here upper bounds for both min_i and min_j are both 4 x vscale, as we know
// that is always the largest value they could take. As if `dim < 4 x vscale`
// then 4 x vscale is an overestimate, and if `dim > 4 x vscale` then the min
// will be clamped to 4 x vscale.

// CHECK: #[[$SCALABLE_BOUND_MAP_1:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @dynamic_size_loop_nest
//       CHECK:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_1]]()[%vscale]
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]], %[[SCALABLE_BOUND]]) : (index, index) -> ()
func.func @dynamic_size_loop_nest(%dim0: index, %dim1: index) {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %dim0 step %c4_vscale {
    %min_i = affine.min #dynamicDim0Map(%i)[%c4_vscale, %dim0]
    scf.for %j = %c0 to %dim1 step %c4_vscale {
      %min_j = affine.min #dynamicDim1Map(%j)[%c4_vscale, %dim1]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB"} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB"} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

// Here the upper bound is just a value + a constant.

// CHECK: #[[$SCALABLE_BOUND_MAP_2:.*]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: @add_to_vscale
//       CHECK:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_2]]()[%vscale]
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]]) : (index) -> ()
func.func @add_to_vscale() {
  %vscale = vector.vscale
  %c8 = arith.constant 8 : index
  %vscale_plus_c8 = arith.addi %vscale, %c8 : index
  %bound = "test.reify_scalable_bound"(%vscale_plus_c8) {type = "UB"} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we know vscale is always 2 so we get a constant upper bound.

// CHECK-LABEL: @vscale_fixed_size
//       CHECK:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   "test.some_use"(%[[C2]]) : (index) -> ()
func.func @vscale_fixed_size() {
  %vscale = vector.vscale
  %bound = "test.reify_scalable_bound"(%vscale) {type = "UB", vscale_min = 2, vscale_max = 2} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we don't know the upper bound (%a is underspecified)

func.func @unknown_bound(%a: index) {
  %vscale = vector.vscale
  %vscale_plus_a = arith.muli %vscale, %a : index
  // expected-error @below{{could not reify bound}}
  %bound = "test.reify_scalable_bound"(%vscale_plus_a) {type = "UB"} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we have two vscale values (that have not been CSE'd), but they should
// still be treated as equivalent.

// CHECK: #[[$SCALABLE_BOUND_MAP_3:.*]] = affine_map<()[s0] -> (s0 * 6)>

// CHECK-LABEL: @duplicate_vscale_values
//       CHECK:   %[[SCALABLE_BOUND:.*]] = affine.apply #[[$SCALABLE_BOUND_MAP_3]]()[%vscale]
//       CHECK:   "test.some_use"(%[[SCALABLE_BOUND]]) : (index) -> ()
func.func @duplicate_vscale_values() {
  %c4 = arith.constant 4 : index
  %vscale_0 = vector.vscale

  %c2 = arith.constant 2 : index
  %vscale_1 = vector.vscale

  %c4_vscale = arith.muli %vscale_0, %c4 : index
  %c2_vscale = arith.muli %vscale_1, %c2 : index
  %add = arith.addi %c2_vscale, %c4_vscale : index

  %bound = "test.reify_scalable_bound"(%add) {type = "UB"} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}
