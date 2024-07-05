// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = generic_reduce_2d_f32
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_native_utils_lib_dir/libmlir_runner_utils%shlibext,%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext

// RUN: %{compile}

// RUN: %{run} | FileCheck %s --check-prefix=F32

func.func @generic_reduce_2d_f32() {
  // 2-D Tensor
  %M = arith.constant 16 : index
  %N = arith.constant 1000 : index
  %c0_f32 = arith.constant 0.0 : f32

  // Allocate the input and output tensors
  %A_alloc = bufferization.alloc_tensor(%M, %N) : tensor<?x?xf32>
  %C_alloc = bufferization.alloc_tensor(%M) : tensor<?xf32>

  // Initialise the tensors
  %pi = arith.constant  3.1416 : f32
  %A_in = linalg.fill ins(%pi : f32) outs(%A_alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %C_in = linalg.fill ins(%c0_f32 : f32) outs(%C_alloc : tensor<?xf32>) -> tensor<?xf32>

  // Reduce
  %C_out = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0)>],
                        iterator_types = ["parallel", "reduction"] }
    ins(%A_in : tensor<?x?xf32>)
    outs(%C_in : tensor<?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>

  // Print and verify the output
  // F32-LABEL: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // F32-NEXT: Unranked Memref {{.*}} rank = 1 offset = 0 sizes = [16] strides = [1] data =
  // F32-NEXT: [3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6,  3141.6]

  %xf = tensor.cast %C_out : tensor<?xf32> to tensor<*xf32>
  call @printMemrefF32(%xf) : (tensor<*xf32>) -> ()

  // F32-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  // A sequence that will tile and vectorise a Reduce Op
  transform.named_sequence @tile_and_vectorize_reduce(%func
    : !transform.op<"func.func"> {transform.readonly}) {

    // Step 0: Get a handle to the reduce Op
    %reduce = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op

    // Step 1: Tile
    %tiled_reduce, %loops:2 = transform.structured.tile_using_for %reduce tile_sizes [1, [4]]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Step 2: Vectorize
    transform.structured.vectorize %tiled_reduce vector_sizes [1, [4]] : !transform.any_op

    // Step 3: Lower vector.multi_reduction
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.op<"func.func">

    transform.yield
  }

  // A sequence that goes over all functions in tis module and applies
  // "tile_and_vectorize_reduce"
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module
        : (!transform.any_op) -> !transform.op<"func.func">

    transform.foreach %funcs : !transform.op<"func.func"> {
      ^bb2(%func : !transform.op<"func.func">):
        transform.include @tile_and_vectorize_reduce failures(propagate)
        (%func) : (!transform.op<"func.func">) -> ()
    }
    transform.yield
  }
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
