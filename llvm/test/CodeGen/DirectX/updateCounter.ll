; RUN: opt -S -dxil-op-lower %s | FileCheck %s


target triple = "dxil-pc-shadermodel6.6-compute"

define void @update_counter_decrement_vector() {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.updateCounter(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i8 -1)
  ret void
}

define void @update_counter_increment_vector() {
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

  call void @llvm.dx.updateCounter(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i8 1)
  ret void
}

define void @update_counter_decrement_scalar() {
  %buffer = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.handle.fromBinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)

  call void @llvm.dx.updateCounter(target("dx.RawBuffer", i8, 0, 0) %buffer, i8 -1)
  ret void
}
