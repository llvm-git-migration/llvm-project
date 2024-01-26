//===-- Passes.td - Mesh transformation definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//

def ShardingPropagation : Pass<"sharding-propagation", "mlir::func::FuncOp"> {
  let summary = "sharding propagation";
  let description = [{
    Propagates sharding information throughout the graph. After this pass, each
    of the operations' operands and results is annotated with a `mesh.shard`
    operation, and the operations themselves are added with sharding option
    attributes.
  }];
  let dependentDialects = [
    "mesh::MeshDialect"
  ];
}

def Spmdization : Pass<"mesh-spmdization", "mlir::func::FuncOp"> {
  let summary = "Partition a function into SPMD form.";
  let description = [{
    This pass fits in right after a pass that annotates the function with
    shardings like the `ShardingPropagation` pass.
    It operates on a fully annotated IR.

    A fully annotated IR required that all ranked tensor operands, results and
    block arguments are annotated with the `mesh.shard` operation.
  
    All direct descendant operations in the function must implement the
    `ShardingInterface` interface or all their ranked tensor operands and
    results must have full replication sharding.

    The input IR must have sharding annotations such that each operation
    that implements `ShardingInterface` can handle during spmdization with
    its `spmdize` method.
    This can be achieved with the `ShardingPropagation` pass.

    If the function has multiple terminating blocks,
    it is the responsibility of the the one who annotates the function with
    shardings to make sure that all returns would be consisted that is,
    have the same sharding.

    Example:
    ```mlir
    mesh.mesh @mesh_1d(shape = 2)

    func.func @f(
      %arg0: tensor<2xi8>
    // CHECK-SAME: -> tensor<2xi8> {
    ) -> tensor<2xi8> {
      %0 = mesh.shard %arg0 to <@mesh_1d, [[]]> : tensor<2xi8>
      %1 = mesh.shard %0 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
      %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
      %3 = mesh.shard %2 to <@mesh_1d, [[0]]> : tensor<2xi8>
      %4 = mesh.shard %3 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
      return %4 : tensor<2xi8>
    }
    ```
    Spmdizing the above would result in 
    * resharding the fully replicated input into splitting it along the only
    tensor axis.
    * Performing the element-wise `abs` operation on each device.
    * Resharding back to full replication with an all-gather.

    ```mlir
    mesh.mesh @mesh_1d(shape = 2)
    func.func @f(%arg0: tensor<2xi8>) -> tensor<2xi8> {
      // Reshard [[]] -> [[0]]
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = mesh.process_multi_index on @mesh_1d axes = [0] : index
      %1 = mesh.mesh_shape @mesh_1d axes = [0] : index
      %2 = arith.remui %c2, %1 : index
      %3 = arith.cmpi eq, %2, %c0 : index
      cf.assert %3, "Sharding a tensor with axis size that is not exactly divisible by the mesh axis size is not supported."
      %4 = arith.divui %c2, %1 : index
      %5 = arith.muli %4, %0 : index
      %extracted_slice = tensor.extract_slice %arg0[%5] [1] [1] : tensor<2xi8> to tensor<1xi8>

      %6 = tosa.abs %extracted_slice : (tensor<1xi8>) -> tensor<1xi8>

      // Reshard [[0]] -> [[]]
      %7 = mesh.all_gather %6 on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>

      return %7 : tensor<2xi8>
    }
    ```
  }];
  let dependentDialects = [
    "mesh::MeshDialect"
  ];
}

#endif // MLIR_DIALECT_MESH_TRANSFORMS_PASSES_TD
