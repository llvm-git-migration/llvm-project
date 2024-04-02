// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(tosa-infer-shapes,tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith))" | \
// RUN: mlir-opt -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s

// Validate that the TOSA lowering for tosa.max_pool2d produces the same results when
// for fully static and fully dynamic inputs.

!Generator = !llvm.ptr

// Utility functions
llvm.func @printI1(i1)
func.func private @rtsrand(index) -> (!Generator)
func.func private @rtrand(!Generator, index) -> (index)
func.func private @rtdrand(!Generator) -> ()

func.func @max_pool_static(%arg0: tensor<1x6x34x62xf32>) -> (tensor<1x6x34x62xf32>) {
  %0 = tosa.max_pool2d %arg0 {
    pad = array<i64: 1, 1, 1, 1>,
    kernel = array<i64: 3, 3>,
    stride = array<i64: 1, 1>
  } : (tensor<1x6x34x62xf32>) -> tensor<1x6x34x62xf32>
  return %0 : tensor<1x6x34x62xf32>
}

func.func @max_pool_dynamic(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>) {
  %0 = tosa.max_pool2d %arg0 {
    pad = array<i64: 1, 1, 1, 1>,
    kernel = array<i64: 3, 3>,
    stride = array<i64: 1, 1>
  } : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// Test harness to compare the results of a fully statically shaped max_pool2d with
// a fully dynamically shaped max_pool2d on the same inputs.
func.func @main() {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50.0 : f32
  %c0 = arith.constant 0 : index

  // Allocate a generator
  %g = func.call @rtsrand(%c0) : (index) ->(!Generator)

  // Randomly generate some inputs
  %A = tensor.generate {
  ^bb0(%x: index, %y: index, %z: index, %w: index):
    %ri0 = func.call @rtrand(%g, %c100) : (!Generator, index) -> (index)

    %asint = arith.index_cast %ri0 : index to i64
    %float = arith.uitofp %asint : i64 to f32
    %result = arith.subf %float, %c50 : f32

    tensor.yield %result : f32
  } : tensor<1x6x34x62xf32>

  %A_dynamic = tensor.cast %A : tensor<1x6x34x62xf32> to tensor<?x?x?x?xf32>

  // Call both static and dynamically sized variants
  %result_static  = func.call @max_pool_static(%A) : (tensor<1x6x34x62xf32>) -> tensor<1x6x34x62xf32>
  %result_dynamic = func.call @max_pool_dynamic(%A_dynamic) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  // Compare the results
  %equal = tosa.equal %result_static, %result_dynamic : (tensor<1x6x34x62xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>

  // Reduce the comparison down to a single boolean
  %eq0 = tosa.reduce_all %equal {axis = 0 : i32} : (tensor<?x?x?x?xi1>) -> tensor<1x?x?x?xi1>
  %eq1 = tosa.reduce_all %eq0 {axis = 1 : i32} : (tensor<1x?x?x?xi1>) -> tensor<1x1x?x?xi1>
  %eq2 = tosa.reduce_all %eq1 {axis = 2 : i32} : (tensor<1x1x?x?xi1>) -> tensor<1x1x1x?xi1>
  %eq3 = tosa.reduce_all %eq2 {axis = 3 : i32} : (tensor<1x1x1x?xi1>) -> tensor<1x1x1x1xi1>

  // Print the single boolean value
  %bool_value = tensor.extract %eq3[%c0, %c0, %c0, %c0] : tensor<1x1x1x1xi1>

  // CHECK: true
  llvm.call @printI1(%bool_value) : (i1) -> ()

  // Cleanup the random generator
  func.call @rtdrand(%g) : (!Generator) -> ()
  return
}

