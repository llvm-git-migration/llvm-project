; RUN: opt -S -mtriple=aarch64-unknown-linux-gnu -O2 < %s | FileCheck %s

define dso_local i64 @func(i64 noundef %0, i64 noundef %1) local_unnamed_addr {
  %3 = add nsw i64 %1, %0
  %4 = mul nsw i64 %3, 3
  ret i64 %4
}

define <vscale x 16 x i1> @testInstCombineSVECmpNE() {
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %2 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8 42, i8 1)
  %3 = tail call i64 @func(i64 noundef 1, i64 noundef 2)
  %4 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %2 , i64 %3)
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %4, <vscale x 16 x i8> zeroinitializer)
  ret <vscale x 16 x i1> %5
  ; CHECK: %4 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 16 x i8> zeroinitializer)
  ; CHECK-NEXT: ret <vscale x 16 x i1> %4
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8, i8)
declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 immarg)
declare <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
