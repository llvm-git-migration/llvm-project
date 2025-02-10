// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda" "-target-feature" "+ptx86" "-target-cpu" "sm_100a" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 "-triple" "nvptx64-nvidia-cuda" "-target-feature" "+ptx86" "-target-cpu" "sm_100a" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

// CHECK: define{{.*}} void @_Z6kernelPiPf(ptr noundef %out, ptr noundef %out_f)
__attribute__((global)) void kernel(int *out, float* out_f) {
  int a = 1;
  unsigned int b = 5;
  float c = 3.0;
  int i = 0;
  int j = 0;

  out[i++] = __nvvm_redux_sync_add(a, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.add

  out[i++] = __nvvm_redux_sync_add(b, 0x01);
  // CHECK: call i32 @llvm.nvvm.redux.sync.add

  out[i++] = __nvvm_redux_sync_min(a, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.min

  out[i++] = __nvvm_redux_sync_umin(b, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.umin

  out[i++] = __nvvm_redux_sync_max(a, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.max

  out[i++] = __nvvm_redux_sync_umax(b, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.umax

  out[i++] = __nvvm_redux_sync_and(a, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.and

  out[i++] = __nvvm_redux_sync_and(b, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.and

  out[i++] = __nvvm_redux_sync_xor(a, 0x10);
  // CHECK: call i32 @llvm.nvvm.redux.sync.xor

  out[i++] = __nvvm_redux_sync_xor(b, 0x01);
  // CHECK: call i32 @llvm.nvvm.redux.sync.xor

  out[i++] = __nvvm_redux_sync_or(a, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.or

  out[i++] = __nvvm_redux_sync_or(b, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.or
  
  out_f[j++] = __nvvm_redux_sync_fmin(c, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin

  out_f[j++] = __nvvm_redux_sync_fmin_abs(c, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.abs

  out_f[j++] = __nvvm_redux_sync_fmin_NaN(c, 0xF0);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.NaN

  out_f[j++] = __nvvm_redux_sync_fmin_abs_NaN(c, 0x0F);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.abs.NaN

  out_f[j++] = __nvvm_redux_sync_fmax(c, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax

  out_f[j++] = __nvvm_redux_sync_fmax_abs(c, 0x01);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.abs

  out_f[j++] = __nvvm_redux_sync_fmax_NaN(c, 0xF1);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.NaN

  out_f[j++] = __nvvm_redux_sync_fmax_abs_NaN(c, 0x10);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.abs.NaN

  // CHECK: ret void
}
