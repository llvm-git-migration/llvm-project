// RUN: mlir-opt --memref-emulate-wide-int="widest-int-supported=32" \
// RUN:   --split-input-file --verify-diagnostics %s

// Make sure we do not crash on unsupported types.

func.func @alloc_i128() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloc' that was explicitly marked illegal}}
  %m = memref.alloc() : memref<4xi128, 1>
  return
}

// -----

func.func @load_i128(%m: memref<4xi128, 1>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'memref.load' that was explicitly marked illegal}}
  %v = memref.load %m[%c0] : memref<4xi128, 1>
  return
}

// -----

func.func @store_i128(%c1: i128, %m: memref<4xi128, 1>) {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'memref.store' that was explicitly marked illegal}}
  memref.store %c1, %m[%c0] : memref<4xi128, 1>
  return
}
