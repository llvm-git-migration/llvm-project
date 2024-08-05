// RUN: not --crash mlir-opt %s --pass-pipeline="builtin.module(func.func(tosa-to-linalg-named))" 2>&1 | FileCheck -check-prefix=CHECK %s

func.func @func1() {
	%arg0 = tensor.empty() : tensor<3x4x5xi32>
	%1110 = arith.constant dense<[3, 0, 1]> : tensor<3xi32>
	%143 = tosa.transpose %arg0, %1110: (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<3x4x5xi32>
	return 
}

// CHECK: permutation must be within input bounds
