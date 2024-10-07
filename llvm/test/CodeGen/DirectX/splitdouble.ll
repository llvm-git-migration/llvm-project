; RUN: opt -S --scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; Make sure DXILOpLowering is correctly generating the dxil op, with and without scalarizer.

; CHECK-LABEL: define noundef i32 @test_scalar
define noundef i32 @test_scalar(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %D)
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.splitdouble = call { i32, i32 } @llvm.dx.splitdouble.i32(double %D)
  %0 = extractvalue { i32, i32 } %hlsl.splitdouble, 0
  %1 = extractvalue { i32, i32 } %hlsl.splitdouble, 1
  %add = add i32 %0, %1
  ret i32 %add
}

;CHECK-LABEL: define noundef <3 x i32> @test_vector
define noundef <3 x i32> @test_vector(<3 x double> noundef %D) local_unnamed_addr {
entry:
  ; CHECK-COUNT-3: [[REG:%.*]] = extractelement <3 x double> %D, i64 {{[0-2]}}
  ; CHECK: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[REG]])
  ; CHECK: [[EXTR0:%.*]] = extractvalue %dx.types.splitdouble [[CALL]], 0
  ; CHECK: [[EXTR1:%.*]] = extractvalue %dx.types.splitdouble [[CALL]], 1
  %hlsl.splitdouble = tail call { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double> %D)
  %0 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.splitdouble, 0
  %1 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.splitdouble, 1
  %add = add <3 x i32> %0, %1
  ret <3 x i32> %add
}
