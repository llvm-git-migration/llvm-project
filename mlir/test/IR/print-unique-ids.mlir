// RUN: mlir-opt -mlir-print-unique-ids %s | FileCheck %s

// CHECK: %arg5
// CHECK: %15
module {
  func.func @uniqueConflicts(%arg0 : memref<i32>, %arg1 : memref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %a = memref.load %arg0[] : memref<i32>
      %b = memref.load %arg1[] : memref<i32>
      %0 = arith.addi %a, %b : i32
      %1 = arith.subi %a, %b : i32
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %a2 = memref.load %arg0[] : memref<i32>
        %b2 = memref.load %arg1[] : memref<i32>
        %2 = arith.addi %a2, %b2 : i32
        %3 = arith.subi %a2, %b2 : i32
        scf.yield
      }
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %a2 = memref.load %arg0[] : memref<i32>
        %b2 = memref.load %arg1[] : memref<i32>
        %2 = arith.addi %a2, %b2 : i32
        %3 = arith.subi %a2, %b2 : i32
        scf.yield
      }
      scf.yield
    }
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %a = memref.load %arg0[] : memref<i32>
      %b = memref.load %arg1[] : memref<i32>
      %0 = arith.addi %a, %b : i32
      %1 = arith.subi %a, %b : i32
      scf.yield
    }
    return
  }
}
