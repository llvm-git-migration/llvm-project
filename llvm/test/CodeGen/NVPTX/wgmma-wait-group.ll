; RUN: llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | FileCheck %s
; RUN: %if ptxas-12.0 %{ llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | %ptxas-verify -arch=sm_90a %}

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.wgmma.wait_group.sync.aligned(i32)

define void @test_wgmma_wait_group_sync_aligned() {
  ; CHECK-LABEL:  test_wgmma_wait_group_sync_aligned(
  ; CHECK:        // %bb.0:
  ; CHECK-NEXT:     wgmma.wait_group.sync.aligned   10;
  ; CHECK-NEXT:     ret;
  call void @llvm.nvvm.wgmma.wait_group.sync.aligned(i32 10)
  ret void
}