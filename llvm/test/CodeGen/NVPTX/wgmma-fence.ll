; RUN: llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | FileCheck %s
; RUN: %if ptxas-12.0 %{ llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | %ptxas-verify -arch=sm_90a %}

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.wgmma.fence.sync.aligned()

define void @test_wgmma_fence_sync_aligned() {
  ; CHECK-LABEL:  test_wgmma_fence_sync_aligned(
  ; CHECK:        // %bb.0:
  ; CHECK-NEXT:     wgmma.fence.sync.aligned;
  ; CHECK-NEXT:     ret;
  call void @llvm.nvvm.wgmma.fence.sync.aligned()
  ret void
}