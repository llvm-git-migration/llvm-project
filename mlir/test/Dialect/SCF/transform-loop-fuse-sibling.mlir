// RUN: mlir-opt %s -transform-interpreter --cse --canonicalize -split-input-file -verify-diagnostics | FileCheck %s

func.func @matmul_fuse_1st_forall_into_2nd(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: scf.forall ([[I:%.*]]) in (4) shared_outs([[S1:%.*]] = [[IN1:%.*]], [[S2:%.*]] = [[IN2:%.*]]) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  // CHECK:   [[T:%.*]] = affine.apply
  // CHECK:   tensor.extract_slice [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT1:%.*]] = linalg.matmul
  // CHECK:   tensor.extract_slice [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   [[OUT2:%.*]] = linalg.matmul
  // CHECK:   scf.forall.in_parallel {
  // CHECK:     tensor.parallel_insert_slice [[OUT1]] into [[S1]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:     tensor.parallel_insert_slice [[OUT2]] into [[S2]][[[T]], 0] [32, 128] [1, 1]
  // CHECK:   }
  // CHECK: }
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_fuse_1st_forall_into_2nd_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // expected-error @below {{user of results of target should be properly dominated by source}}
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %out1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop1 into %loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_fuse_2nd_for_into_1st_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<32xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %first_loop, %second_loop = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}



// -----
func.func @matmul_fuse_2nd_forall_into_1st_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{values used inside regions of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A, %out1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_fuse_2nd_forall_into_1st_err(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  // expected-error @below {{operands of target should be properly dominated by source}}
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_forall %mm1 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_forall %mm2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_loop = transform.loop.fuse_sibling %loop2 into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @matmul_fuse_1st_for_into_2nd([[A:%.*]]: {{.*}}, [[B1:%.*]]: {{.*}}, [[B2:%.*]]: {{.*}} {{.*}}
func.func @matmul_fuse_1st_for_into_2nd(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C32:%.*]] = arith.constant 32 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.*]] = tensor.empty() : tensor<128x128xf32>
  // CHECK-DAG: [[BUF:%.*]] = linalg.fill ins([[ZERO]] : {{.*}}) outs([[EMPTY]] : {{.*}}) {{.*}}
  // CHECK: [[RST:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C32]] iter_args([[IA0:%.*]] = [[BUF]], [[IA1:%.*]] = [[BUF]]) {{.*}}
  // CHECK-DAG:   [[ASLICE:%.*]] = tensor.extract_slice [[A]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK-DAG:   [[SLICE0:%.*]] = tensor.extract_slice [[IA0]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK:       [[OUT1:%.*]] = linalg.matmul ins([[ASLICE]], [[B1]] : {{.*}}) outs([[SLICE0]]
  // CHECK-NEXT:  [[INS0:%.*]] = tensor.insert_slice [[OUT1]] into [[IA0]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK-DAG:   [[SLICE1:%.*]] = tensor.extract_slice [[IA1]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK:       [[OUT2:%.*]] = linalg.matmul ins([[ASLICE]], [[B2]] : {{.*}}) outs([[SLICE1]]
  // CHECK-NEXT:  [[INS1:%.*]] = tensor.insert_slice [[OUT2]] into [[IA1]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK: scf.yield [[INS0]], [[INS1]] : {{.*}}
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: return [[RST]]#0, [[RST]]#1 : {{.*}}
  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_for %mm1 [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_for %mm2 [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)
    %cse_func = transform.apply_registered_pass "cse" to %func : (!transform.any_op) -> (!transform.any_op)
    %for_loops = transform.structured.match ops{["scf.for"]} in %cse_func : (!transform.any_op) -> (!transform.any_op)
    %for_loop1, %for_loop2 = transform.split_handle %for_loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_loop = transform.loop.fuse_sibling %for_loop2 into %for_loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func.func @matmul_fuse_2nd_for_into_1st([[A:%.*]]: {{.*}}, [[B1:%.*]]: {{.*}}, [[B2:%.*]]: {{.*}} {{.*}}
func.func @matmul_fuse_2nd_for_into_1st(%A : tensor<128x128xf32>, %B1 : tensor<128x128xf32>, %B2 : tensor<128x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %zero = arith.constant 0.0 : f32
  %out_alloc = tensor.empty() : tensor<128x128xf32>
  %out = linalg.fill ins(%zero : f32) outs(%out_alloc : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C32:%.*]] = arith.constant 32 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[EMPTY:%.*]] = tensor.empty() : tensor<128x128xf32>
  // CHECK-DAG: [[BUF:%.*]] = linalg.fill ins([[ZERO]] : {{.*}}) outs([[EMPTY]] : {{.*}}) {{.*}}
  // CHECK: [[RST:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C32]] iter_args([[IA0:%.*]] = [[BUF]], [[IA1:%.*]] = [[BUF]]) {{.*}}
  // CHECK-DAG:   [[ASLICE:%.*]] = tensor.extract_slice [[A]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK-DAG:   [[SLICE0:%.*]] = tensor.extract_slice [[IA0]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK:       [[OUT1:%.*]] = linalg.matmul ins([[ASLICE]], [[B2]] : {{.*}}) outs([[SLICE0]]
  // CHECK-NEXT:  [[INS0:%.*]] = tensor.insert_slice [[OUT1]] into [[IA0]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK-DAG:   [[SLICE1:%.*]] = tensor.extract_slice [[IA1]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK:       [[OUT2:%.*]] = linalg.matmul ins([[ASLICE]], [[B1]] : {{.*}}) outs([[SLICE1]]
  // CHECK-NEXT:  [[INS1:%.*]] = tensor.insert_slice [[OUT2]] into [[IA1]][[[IV]], 0] [32, 128] [1, 1]
  // CHECK: scf.yield [[INS0]], [[INS1]] : {{.*}}
  %out1 = linalg.matmul ins(%A, %B1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>
  %out2 = linalg.matmul ins(%A, %B2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%out : tensor<128x128xf32>) -> tensor<128x128xf32>

  // CHECK: return [[RST]]#1, [[RST]]#0 : {{.*}}
  func.return %out1, %out2 : tensor<128x128xf32>, tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op : !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)

    %mm1, %mm2 = transform.split_handle %matched : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_mm1, %loop1 = transform.structured.tile_using_for %mm1 [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_mm2, %loop2 = transform.structured.tile_using_for %mm2 [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> (!transform.any_op)
    %cse_func = transform.apply_registered_pass "cse" to %func : (!transform.any_op) -> (!transform.any_op)
    %for_loops = transform.structured.match ops{["scf.for"]} in %cse_func : (!transform.any_op) -> (!transform.any_op)
    %for_loop1, %for_loop2 = transform.split_handle %for_loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_loop = transform.loop.fuse_sibling %for_loop1 into %for_loop2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// transform.loop.fuse_sibling used to silently fail on the following due to a bug in the dominance check

// CHECK: func.func @no_dominance_bug([[A:%.*]]: {{.*}}, [[B:%.*]]: {{.*}}
func.func @no_dominance_bug(%arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C16:%.*]] = arith.constant 16 : index
  // CHECK-DAG: [[C128:%.*]] = arith.constant 128 : index
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: [[RST:%.*]]:2 = scf.for [[IV:%.*]] = [[C0]] to [[C128]] step [[C16]] iter_args([[IB0:%.*]] = [[B]], [[IB1:%.*]] = [[B]]) {{.*}}
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[ASLICE:%.*]] = vector.transfer_read [[A]][[[IV]]], [[ZERO]]
  // CHECK-DAG:   [[SLICE0:%.*]] = vector.transfer_read [[IB0]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT1:%.*]] = arith.addf [[SLICE0]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT0:%.*]] = vector.transfer_write [[OUT1]], [[IB0]][[[IV]]]
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
  // CHECK-DAG:   [[SLICE1:%.*]] = vector.transfer_read [[IB1]][[[IV]]], [[ZERO]]
  // CHECK:       [[OUT2:%.*]] = arith.addf [[SLICE1]], [[ASLICE]]
  // CHECK-NEXT:  [[WRT1:%.*]] = vector.transfer_write [[OUT2]], [[IB1]][[[IV]]]
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<16xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
  // CHECK: scf.yield [[WRT0]], [[WRT1]] : {{.*}}
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}

// -----

func.func @dominance_check_violation(%arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c128 step %c16 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
    %2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<16xf32>
    %5 = arith.addf %3, %2 : vector<16xf32>
    %6 = vector.transfer_write %5, %arg4[%arg3] {in_bounds = [true]} : vector<16xf32>, tensor<128xf32>
    scf.yield %6 : tensor<128xf32>
  }
  %c32 = arith.constant 32 : index // When fusing the subsequent loop into the prior loop, this value is used before its defined.
  // expected-error @below {{operands of target should be properly dominated by source}}
  %dup1 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %arg2) -> (tensor<128xf32>) {
    %dup2 = vector.transfer_read %arg1[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup3 = vector.transfer_read %arg4[%arg3], %cst {in_bounds = [true]} : tensor<128xf32>, vector<32xf32>
    %dup5 = arith.addf %dup3, %dup2 : vector<32xf32>
    %dup6 = vector.transfer_write %dup5, %arg4[%arg3] {in_bounds = [true]} : vector<32xf32>, tensor<128xf32>
    scf.yield %dup6 : tensor<128xf32>
  }
  return %1, %dup1 : tensor<128xf32>, tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %for:2 = transform.split_handle %0 :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused = transform.loop.fuse_sibling %for#1 into %for#0 : (!transform.any_op,!transform.any_op) ->  !transform.any_op
    transform.yield
  }
}
