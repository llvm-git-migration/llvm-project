// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --verify-diagnostics --split-input-file %s

func.func @negative_memref_subview_non_contiguous(%idx : index) -> i4 {
  %c0 = arith.constant 0 : index
  %arr = memref.alloc() : memref<40x40xi4>
  // expected-error @+1 {{failed to legalize operation 'memref.subview' that was explicitly marked illegal}}
  %subview = memref.subview %arr[%idx, 0] [4, 8] [1, 1] : memref<40x40xi4> to memref<4x8xi4, strided<[40, 1], offset:?>>
  %ld = memref.load %subview[%c0, %c0] : memref<4x8xi4, strided<[40, 1], offset:?>>
  return %ld : i4
}

// -----

func.func @alloc_non_contiguous() {
  // expected-error @+1 {{failed to legalize operation 'memref.alloc' that was explicitly marked illegal}}
  %arr = memref.alloc() : memref<8x8xi4, strided<[1, 8]>>
  return
}

// -----

// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @argument_non_contiguous(%arg0 : memref<8x8xi4, strided<[1, 8]>>) {
  return
}
