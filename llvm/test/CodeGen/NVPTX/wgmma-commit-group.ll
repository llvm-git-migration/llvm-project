; RUN: llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | FileCheck %s
; RUN: %if ptxas-12.0 %{ llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80 | %ptxas-verify -arch=sm_90a %}

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.wgmma.commit_group.sync.aligned()

define void @test_wgmma_commit_group_sync_aligned() {
  ; CHECK-LABEL:  test_wgmma_commit_group_sync_aligned(
  ; CHECK:        // %bb.0:
  ; CHECK-NEXT:     wgmma.commit_group.sync.aligned;
  ; CHECK-NEXT:     ret;
  call void @llvm.nvvm.wgmma.commit_group.sync.aligned()
  ret void
}