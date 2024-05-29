// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// Outlined functions:
//
// CHECK: func @foo(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK: func @foo[[$SUFFIX:.+]](%{{.+}}, %{{.+}}, %{{.+}})
// CHECK:   scf.for
// CHECK:     arith.addi
//
// CHECK-LABEL: @loop_outline_op
func.func @loop_outline_op(%arg0: index, %arg1: index, %arg2: index) {
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // CHECK:   scf.execute_region
  // CHECK:     func.call @foo
  scf.for %i = %arg0 to %arg1 step %arg2 {
    scf.for %j = %arg0 to %arg1 step %arg2 {
      arith.addi %i, %j : index
    }
  }
  // CHECK: scf.execute_region
  // CHECK-NOT: scf.for
  // CHECK:   func.call @foo[[$SUFFIX]]
  scf.for %j = %arg0 to %arg1 step %arg2 {
    arith.addi %j, %j : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // CHECK: = transform.loop.outline %{{.*}}
    transform.loop.outline %1 {func_name = "foo"} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_peel_op
func.func @loop_peel_op() {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[C41:.+]] = arith.constant 41
  // CHECK: %[[C5:.+]] = arith.constant 5
  // CHECK: %[[C40:.+]] = arith.constant 40
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C40]] step %[[C5]]
  // CHECK:   arith.addi
  // CHECK: scf.for %{{.+}} = %[[C40]] to %[[C41]] step %[[C5]]
  // CHECK:   arith.addi
  %0 = arith.constant 0 : index
  %1 = arith.constant 41 : index
  %2 = arith.constant 5 : index
  // expected-remark @below {{main loop}}
  // expected-remark @below {{remainder loop}}
  scf.for %i = %0 to %1 step %2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    // Make sure 
    transform.debug.emit_remark_at %main_loop, "main loop" : !transform.op<"scf.for">
    transform.debug.emit_remark_at %remainder, "remainder loop" : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_peel_first_iter_op
func.func @loop_peel_first_iter_op() {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[C41:.+]] = arith.constant 41
  // CHECK: %[[C5:.+]] = arith.constant 5
  // CHECK: %[[C5_0:.+]] = arith.constant 5
  // CHECK: scf.for %{{.+}} = %[[C0]] to %[[C5_0]] step %[[C5]]
  // CHECK:   arith.addi
  // CHECK: scf.for %{{.+}} = %[[C5_0]] to %[[C41]] step %[[C5]]
  // CHECK:   arith.addi
  %0 = arith.constant 0 : index
  %1 = arith.constant 41 : index
  %2 = arith.constant 5 : index
  scf.for %i = %0 to %1 step %2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %main_loop, %remainder = transform.loop.peel %1 {peel_front = true} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.yield
  }
}

// -----

func.func @loop_pipeline_op(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  // CHECK: memref.load %[[MEMREF:.+]][%{{.+}}]
  // CHECK: memref.load %[[MEMREF]]
  // CHECK: arith.addf
  // CHECK: scf.for
  // CHECK:   memref.load
  // CHECK:   arith.addf
  // CHECK:   memref.store
  // CHECK: arith.addf
  // CHECK: memref.store
  // CHECK: memref.store
  // expected-remark @below {{transformed}}
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf : f32
    memref.store %A1_elem, %result[%i0] : memref<?xf32>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %2 = transform.loop.pipeline %1 : (!transform.op<"scf.for">) -> !transform.any_op
    // Verify that the returned handle is usable.
    transform.debug.emit_remark_at %2, "transformed" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_op
func.func @loop_unroll_op() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  // CHECK: scf.for %[[I:.+]] =
  scf.for %i = %c0 to %c42 step %c5 {
    // CHECK-COUNT-4: arith.addi %[[I]]
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_1:.*]] = arith.constant 40 : index
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK:           %[[FACTOR:.*]] = arith.constant 4 : index
  // CHECK:           %[[STEP:.*]] = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c2 = arith.constant 2 : index
  // CHECK:           scf.for %[[VAL_5:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[STEP]] {
  scf.for %i = %c0 to %c40 step %c2 {
  // CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : index
  // CHECK:             %[[VAL_7:.*]] = arith.constant 2 : index
  // CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_5]], %[[VAL_7]] : index
  // CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_8]] : index
  // CHECK:             %[[VAL_10:.*]] = arith.constant 4 : index
  // CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_5]], %[[VAL_10]] : index
  // CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_11]] : index
  // CHECK:             %[[VAL_13:.*]] = arith.constant 6 : index
  // CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_5]], %[[VAL_13]] : index
  // CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_14]] : index
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
// CHECK:       %[[VAL_0:.*]]: memref<96x128xi8, 3>, %[[VAL_1:.*]]: memref<128xi8, 3>) {
func.func private @loop_unroll_and_jam_op(%arg0: memref<96x128xi8, 3>, %arg1: memref<128xi8, 3>) {
  // CHECK:           %[[VAL_2:.*]] = arith.constant 96 : index
  // CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_4:.*]] = arith.constant 128 : index
  // CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
  %c96 = arith.constant 96 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  // CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_7]] {
	scf.for %arg2 = %c0 to %c128 step %c1 {
    // CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_8]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_10:.*]] = arith.constant 1 : index
    // CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : index
    // CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_11]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_13:.*]] = arith.constant 2 : index
    // CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_8]], %[[VAL_13]] : index
    // CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_14]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_16:.*]] = arith.constant 3 : index
    // CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_8]], %[[VAL_16]] : index
    // CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_17]]] : memref<128xi8, 3>
	  %3 = memref.load %arg1[%arg2] : memref<128xi8, 3>
    // CHECK:             %[[VAL_19:.*]]:4 = scf.for %[[VAL_20:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_9]], %[[VAL_22:.*]] = %[[VAL_12]], %[[VAL_23:.*]] = %[[VAL_15]], %[[VAL_24:.*]] = %[[VAL_18]]) -> (i8, i8, i8, i8) {
	  %sum = scf.for %arg3 = %c0 to %c96 step %c1 iter_args(%does_not_alias_aggregated = %3) -> (i8) {
    // CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_8]]] : memref<96x128xi8, 3>
    // CHECK:               %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_9]] : i8
    // CHECK:               %[[VAL_27:.*]] = arith.constant 1 : index
    // CHECK:               %[[VAL_28:.*]] = arith.addi %[[VAL_8]], %[[VAL_27]] : index
    // CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_28]]] : memref<96x128xi8, 3>
    // CHECK:               %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_12]] : i8
    // CHECK:               %[[VAL_31:.*]] = arith.constant 2 : index
    // CHECK:               %[[VAL_32:.*]] = arith.addi %[[VAL_8]], %[[VAL_31]] : index
    // CHECK:               %[[VAL_33:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_32]]] : memref<96x128xi8, 3>
    // CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_15]] : i8
    // CHECK:               %[[VAL_35:.*]] = arith.constant 3 : index
    // CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_8]], %[[VAL_35]] : index
    // CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_36]]] : memref<96x128xi8, 3>
    // CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_18]] : i8
		%2 = memref.load %arg0[%arg3, %arg2] : memref<96x128xi8, 3>
		%4 = arith.addi %2, %3 : i8
    // CHECK:               scf.yield %[[VAL_26]], %[[VAL_30]], %[[VAL_34]], %[[VAL_38]] : i8, i8, i8, i8
		scf.yield %4 : i8
	  }
	  memref.store %sum, %arg1[%arg2] : memref<128xi8, 3>
    // CHECK:             memref.store %[[VAL_39:.*]]#0, %[[VAL_1]]{{\[}}%[[VAL_8]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_40:.*]] = arith.constant 1 : index
    // CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_8]], %[[VAL_40]] : index
    // CHECK:             memref.store %[[VAL_39]]#1, %[[VAL_1]]{{\[}}%[[VAL_41]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_42:.*]] = arith.constant 2 : index
    // CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_8]], %[[VAL_42]] : index
    // CHECK:             memref.store %[[VAL_39]]#2, %[[VAL_1]]{{\[}}%[[VAL_43]]] : memref<128xi8, 3>
    // CHECK:             %[[VAL_44:.*]] = arith.constant 3 : index
    // CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_8]], %[[VAL_44]] : index
    // CHECK:             memref.store %[[VAL_39]]#3, %[[VAL_1]]{{\[}}%[[VAL_45]]] : memref<128xi8, 3>
	}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.store"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_1:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : index
  // CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
  // CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_0]], %[[VAL_6]] : index
  // CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_7]] : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c4 step %c2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
func.func @loop_unroll_and_jam_op() {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_1:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  // CHECK:           scf.for %[[VAL_3:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] {
  scf.for %i = %c0 to %c4 step %c2 {
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll_and_jam %1 { factor = 2 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @loop_unroll_and_jam_op
// CHECK:  %[[VAL_0:.*]]: memref<21x30xf32, 1>, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) {
func.func @loop_unroll_and_jam_op(%arg0: memref<21x30xf32, 1>, %init : f32, %init1 : f32) {
  // CHECK:           %[[VAL_3:.*]] = arith.constant 20 : index
  // CHECK:           %[[VAL_4:.*]]:2 = affine.for %[[VAL_5:.*]] = 0 to 20 step 2 iter_args(%[[VAL_6:.*]] = %[[VAL_1]], %[[VAL_7:.*]] = %[[VAL_1]]) -> (f32, f32) {
  %0 = affine.for %arg3 = 0 to 21 iter_args(%arg4 = %init) -> (f32) {
    // CHECK:             %[[VAL_8:.*]]:2 = affine.for %[[VAL_9:.*]] = 0 to 30 iter_args(%[[VAL_10:.*]] = %[[VAL_2]], %[[VAL_11:.*]] = %[[VAL_2]]) -> (f32, f32) {
    %1 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %init1) -> (f32) {
      // CHECK:               %[[VAL_12:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_5]], %[[VAL_9]]] : memref<21x30xf32, 1>
      // CHECK:               %[[VAL_13:.*]] = arith.addf %[[VAL_10]], %[[VAL_12]] : f32
      // CHECK:               %[[VAL_14:.*]] = affine.apply #map(%[[VAL_5]])
      // CHECK:               %[[VAL_15:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_9]]] : memref<21x30xf32, 1>
      // CHECK:               %[[VAL_16:.*]] = arith.addf %[[VAL_11]], %[[VAL_15]] : f32
      // CHECK:               affine.yield %[[VAL_13]], %[[VAL_16]] : f32, f32
      %3 = affine.load %arg0[%arg3, %arg5] : memref<21x30xf32, 1>
      %4 = arith.addf %arg6, %3 : f32
      affine.yield %4 : f32
    }
    // CHECK:             %[[VAL_17:.*]] = arith.mulf %[[VAL_6]], %[[VAL_18:.*]]#0 : f32
    // CHECK:             %[[VAL_19:.*]] = affine.apply #map(%[[VAL_5]])
    // CHECK:             %[[VAL_20:.*]] = arith.mulf %[[VAL_7]], %[[VAL_18]]#1 : f32
    // CHECK:             affine.yield %[[VAL_17]], %[[VAL_20]] : f32, f32
    // CHECK:           }
    // CHECK:           %[[VAL_21:.*]] = arith.mulf %[[VAL_22:.*]]#0, %[[VAL_22]]#1 : f32
    // CHECK:           %[[VAL_23:.*]] = affine.for %[[VAL_24:.*]] = 0 to 30 iter_args(%[[VAL_25:.*]] = %[[VAL_2]]) -> (f32) {
    // CHECK:             %[[VAL_26:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_24]]] : memref<21x30xf32, 1>
    // CHECK:             %[[VAL_27:.*]] = arith.addf %[[VAL_25]], %[[VAL_26]] : f32
    // CHECK:             affine.yield %[[VAL_27]] : f32
    %2 = arith.mulf %arg4, %1 : f32
    affine.yield %2 : f32
  }
  // CHECK:           %[[VAL_28:.*]] = arith.mulf %[[VAL_21]], %[[VAL_29:.*]] : f32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.get_parent_op %1 {op_name = "affine.for"} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.loop.unroll_and_jam %2 { factor = 2 } : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @loop_unroll_op() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  // CHECK: affine.for %[[I:.+]] =
  // expected-remark @below {{affine for loop}}
  affine.for %i = %c0 to %c42 {
    // CHECK-COUNT-4: arith.addi
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    transform.debug.emit_remark_at %1, "affine for loop" : !transform.op<"affine.for">
    transform.loop.unroll %1 { factor = 4, affine = true } : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @test_mixed_loops() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  scf.for %j = %c0 to %c42 step %c5 {
    // CHECK: affine.for %[[I:.+]] =
    // expected-remark @below {{affine for loop}}
    affine.for %i = %c0 to %c42 {
      // CHECK-COUNT-4: arith.addi
      arith.addi %i, %i : index
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    transform.debug.emit_remark_at %1, "affine for loop" : !transform.op<"affine.for">
    transform.loop.unroll %1 { factor = 4 } : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_promote_if_one_iteration(
//   CHECK-NOT:   scf.for
//       CHECK:   %[[r:.*]] = "test.foo"
//       CHECK:   return %[[r]]
func.func @test_promote_if_one_iteration(%a: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = scf.for %j = %c0 to %c1 step %c1 iter_args(%arg0 = %a) -> index {
    %1 = "test.foo"(%a) : (index) -> (index)
    scf.yield %1 : index
  }
  return %0 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.loop.promote_if_one_iteration %0 : !transform.any_op
    transform.yield
  }
}


// -----

// CHECK-LABEL: func @test_structural_conversion_patterns(
// CHECK: scf.for {{.*}} -> (memref<f32>) {

func.func @test_structural_conversion_patterns(%a: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.for %j = %c0 to %c10 step %c1 iter_args(%arg0 = %a) -> tensor<f32> {
    %1 = "test.foo"(%arg0) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %0 {
      transform.apply_conversion_patterns.scf.structural_conversions
    } with type_converter {
      transform.apply_conversion_patterns.transform.test_type_converter
    } {  partial_conversion  } : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @coalesce_i32_loops(

// This test checks for loop coalescing success for non-index loop boundaries and step type
func.func @coalesce_i32_loops() {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 128 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.constant 64 : i32
  // CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
  // CHECK: scf.for %[[ARG0:.*]] = %[[C0_I32]] to {{.*}} step %[[C1_I32]]  : i32
  scf.for %i = %0 to %1 step %2 : i32 {
    scf.for %j = %0 to %3 step %2 : i32 {
      arith.addi %i, %j : i32
    }
  } {coalesce}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
