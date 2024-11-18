// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @alloc_foo_1(!llvm.ptr)
llvm.func @dealloc_foo_1(!llvm.ptr)

omp.private {type = private} @box.heap_privatizer : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>  : (i32) -> !llvm.ptr
  llvm.call @alloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield(%7 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @dealloc_foo_1(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

llvm.func @target_allocatable_(%arg0: !llvm.ptr {fir.bindc_name = "lb"}, %arg1: !llvm.ptr {fir.bindc_name = "ub"}, %arg2: !llvm.ptr {fir.bindc_name = "l"}) attributes {fir.internal_name = "_QPtarget_allocatable"} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %4 x f32 {bindc_name = "real_var"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(1 : i64) : i64
  %7 = llvm.alloca %6 x i32 {bindc_name = "mapped_var"} : (i64) -> !llvm.ptr
  %8 = llvm.mlir.constant(1 : i64) : i64
  %9 = llvm.alloca %8 x !llvm.struct<(f32, f32)> {bindc_name = "comp_var"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %12 = llvm.mlir.constant(1 : i64) : i64
  %13 = llvm.alloca %12 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "alloc_var"} : (i64) -> !llvm.ptr
  %14 = llvm.mlir.constant(0 : index) : i64
  %15 = llvm.mlir.constant(1 : index) : i64
  %16 = llvm.mlir.constant(0 : i64) : i64
  %17 = llvm.mlir.zero : !llvm.ptr
  %18 = llvm.mlir.constant(9 : i32) : i32
  %19 = llvm.mlir.zero : !llvm.ptr
  %20 = llvm.getelementptr %19[1] : (!llvm.ptr) -> !llvm.ptr, i32
  %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
  %22 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %23 = llvm.insertvalue %21, %22[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %24 = llvm.mlir.constant(20240719 : i32) : i32
  %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %26 = llvm.mlir.constant(0 : i32) : i32
  %27 = llvm.trunc %26 : i32 to i8
  %28 = llvm.insertvalue %27, %25[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %29 = llvm.trunc %18 : i32 to i8
  %30 = llvm.insertvalue %29, %28[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %31 = llvm.mlir.constant(2 : i32) : i32
  %32 = llvm.trunc %31 : i32 to i8
  %33 = llvm.insertvalue %32, %30[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %34 = llvm.mlir.constant(0 : i32) : i32
  %35 = llvm.trunc %34 : i32 to i8
  %36 = llvm.insertvalue %35, %33[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %37, %11 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %38 = llvm.load %11 : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  llvm.store %38, %13 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
  %39 = llvm.load %arg2 : !llvm.ptr -> i64
  %40 = llvm.icmp "sgt" %39, %16 : i64
  %41 = llvm.select %40, %39, %16 : i1, i64
  %42 = llvm.mlir.constant(1 : i64) : i64
  %43 = llvm.alloca %41 x i8 {bindc_name = "char_var"} : (i64) -> !llvm.ptr
  %44 = llvm.load %arg0 : !llvm.ptr -> i64
  %45 = llvm.load %arg1 : !llvm.ptr -> i64
  %46 = llvm.sub %45, %44 : i64
  %47 = llvm.add %46, %15 : i64
  %48 = llvm.icmp "sgt" %47, %14 : i64
  %49 = llvm.select %48, %47, %14 : i1, i64
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.mul %49, %50 : i64
  %52 = llvm.alloca %51 x f32 {bindc_name = "real_arr"} : (i64) -> !llvm.ptr
  %53 = omp.map.info var_ptr(%7 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "mapped_var"}
  %54 = omp.map.info var_ptr(%13 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) -> !llvm.ptr
  omp.target map_entries(%53 -> %arg3, %54 -> %arg4 : !llvm.ptr, !llvm.ptr) private(@box.heap_privatizer %13 -> %arg5 [map_idx=1] : !llvm.ptr) {
    llvm.call @use_private_var(%arg5) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

llvm.func @use_private_var(!llvm.ptr) -> ()

llvm.func @_FortranAAssign(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<()> attributes {fir.runtime, sym_visibility = "private"}

// The first set of checks ensure that we are calling the offloaded function
// with the right arguments, especially the second argument which needs to
// be a memory reference to the descriptor for the privatized allocatable
// CHECK: define void @target_allocatable_
// CHECK-NOT: define internal void
// CHECK: %[[DESC_ALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1
// CHECK: call void @__omp_offloading_[[OFFLOADED_FUNCTION:.*]](ptr {{[^,]+}},
// CHECK-SAME: ptr %[[DESC_ALLOC]])

// The second set of checks ensure that to allocate memory for the
// allocatable, we are, in fact, using the memory reference of the descriptor
// passed as the second argument to the offloaded function.
// CHECK: define internal void @__omp_offloading_[[OFFLOADED_FUNCTION]]
// CHECK-SAME: (ptr {{[^,]+}}, ptr %[[DESCRIPTOR_ARG:.*]]) {
// CHECK: %[[DESC_TO_DEALLOC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }
// CHECK: call void @alloc_foo_1(ptr %[[DESCRIPTOR_ARG]])


// CHECK: call void @use_private_var(ptr %[[DESC_TO_DEALLOC]]

// Now, check the deallocation of the private var.
// CHECK:  call void @dealloc_foo_1(ptr %[[DESC_TO_DEALLOC]])
