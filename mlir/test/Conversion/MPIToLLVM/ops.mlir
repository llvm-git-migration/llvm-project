// RUN: mlir-opt -convert-mpi-to-llvm %s | FileCheck %s

module {
// CHECK:  llvm.func @MPI_Finalize() -> i32
// CHECK:  llvm.func @MPI_Comm_rank(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:  llvm.mlir.global external @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.struct<"MPI_ABI_Comm", opaque>
// CHECK:  llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32

  func.func @mpi_test(%arg0: memref<100xf32>) {
    %0 = mpi.init : !mpi.retval
// CHECK: %0 = llvm.mlir.zero : !llvm.ptr
// CHECK: %1 = llvm.call @MPI_Init(%0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %2 = builtin.unrealized_conversion_cast %1 : i32 to !mpi.retval

    %retval, %rank = mpi.comm_rank : !mpi.retval, i32
// CHECK: %3 = llvm.mlir.constant(1 : i32) : i32
// CHECK: %4 = llvm.alloca %3 x i32 : (i32) -> !llvm.ptr
// CHECK: %5 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CHECK: %6 = llvm.call @MPI_Comm_rank(%5, %4) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %7 = llvm.load %4 : !llvm.ptr -> i32
// CHECK: %8 = builtin.unrealized_conversion_cast %6 : i32 to !mpi.retval

    mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    %3 = mpi.finalize : !mpi.retval
// CHECK: %11 = llvm.call @MPI_Finalize() : () -> i32

    %4 = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    %5 = mpi.error_class %0 : !mpi.retval
    return
  }
}
