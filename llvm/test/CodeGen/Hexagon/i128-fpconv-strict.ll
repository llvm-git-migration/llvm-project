; RUN: llc < %s -mtriple=hexagon-unknown-CHECK-musl \
; RUN:      | FileCheck %s -check-prefix=CHECK

define i64 @double_to_i128(double %d) nounwind strictfp {
; CHECK-LABEL: double_to_i128:
; CHECK:       // %bb.0:
; CHECK:          call __fixdfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptosi.i128.f64(double %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @double_to_ui128(double %d) nounwind strictfp {
; CHECK-LABEL: double_to_ui128:
; CHECK:       // %bb.0:
; CHECK:          call __fixunsdfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptoui.i128.f64(double %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @float_to_i128(float %d) nounwind strictfp {
; CHECK-LABEL: float_to_i128:
; CHECK:       // %bb.0:
; CHECK:          call __fixsfti
; CHECK:          dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptosi.i128.f32(float %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @float_to_ui128(float %d) nounwind strictfp {
; CHECK-LABEL: float_to_ui128:
; CHECK:       // %bb.0:
; CHECK:         call __fixunssfti
; CHECK:         dealloc_return
  %1 = tail call i128 @llvm.experimental.constrained.fptoui.i128.f32(float %d, metadata !"fpexcept.strict")
  %2 = trunc i128 %1 to i64
  ret i64 %2
}

define i64 @longdouble_to_i128(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: longdouble_to_i128:
; CHECK:       // %bb.0:
; CHECK:         call __fixxfti
; CHECK:         dealloc_return
  %2 = load x86_fp80, ptr %0, align 16
  %3 = tail call i128 @llvm.experimental.constrained.fptosi.i128.f80(x86_fp80 %2, metadata !"fpexcept.strict")
  %4 = trunc i128 %3 to i64
  ret i64 %4
}

define i64 @longdouble_to_ui128(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: longdouble_to_ui128:
; CHECK:       // %bb.0:
; CHECK:         call __fixunsxfti
; CHECK:         dealloc_return
  %2 = load x86_fp80, ptr %0, align 16
  %3 = tail call i128 @llvm.experimental.constrained.fptoui.i128.f80(x86_fp80 %2, metadata !"fpexcept.strict")
  %4 = trunc i128 %3 to i64
  ret i64 %4
}

define double @i128_to_double(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: i128_to_double:
; CHECK:       // %bb.0:
; CHECK:         call __floattidf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call double @llvm.experimental.constrained.sitofp.f64.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret double %3
}

define double @ui128_to_double(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: ui128_to_double:
; CHECK:       // %bb.0:
; CHECK:         call __floatuntidf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call double @llvm.experimental.constrained.uitofp.f64.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret double %3
}

define float @i128_to_float(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: i128_to_float:
; CHECK:       // %bb.0:
; CHECK:         call __floattisf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call float @llvm.experimental.constrained.sitofp.f32.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %3
}

define float @ui128_to_float(ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: ui128_to_float:
; CHECK:       // %bb.0:
; CHECK:         call __floatuntisf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call float @llvm.experimental.constrained.uitofp.f32.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret float %3
}

define void @i128_to_longdouble(ptr noalias nocapture sret(x86_fp80) align 16 %agg.result, ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: i128_to_longdouble:
; CHECK:       // %bb.0:
; CHECK:         call __floattixf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call x86_fp80 @llvm.experimental.constrained.sitofp.f80.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  store x86_fp80 %3, ptr %agg.result, align 16
  ret void
}

define void @ui128_to_longdouble(ptr noalias nocapture sret(x86_fp80) align 16 %agg.result, ptr nocapture readonly %0) nounwind strictfp {
; CHECK-LABEL: ui128_to_longdouble:
; CHECK:       // %bb.0:
; CHECK:         call __floatuntixf
; CHECK:         dealloc_return
  %2 = load i128, ptr %0, align 16
  %3 = tail call x86_fp80 @llvm.experimental.constrained.uitofp.f80.i128(i128 %2, metadata !"round.dynamic", metadata !"fpexcept.strict")
  store x86_fp80 %3, ptr %agg.result, align 16
  ret void
}

declare i128 @llvm.experimental.constrained.fptosi.i128.f64(double, metadata)
declare i128 @llvm.experimental.constrained.fptoui.i128.f64(double, metadata)
declare i128 @llvm.experimental.constrained.fptosi.i128.f32(float, metadata)
declare i128 @llvm.experimental.constrained.fptoui.i128.f32(float, metadata)
declare i128 @llvm.experimental.constrained.fptosi.i128.f80(x86_fp80, metadata)
declare i128 @llvm.experimental.constrained.fptoui.i128.f80(x86_fp80, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i128(i128, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.sitofp.f32.i128(i128, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i128(i128, metadata, metadata)
declare x86_fp80 @llvm.experimental.constrained.sitofp.f80.i128(i128, metadata, metadata)
declare x86_fp80 @llvm.experimental.constrained.uitofp.f80.i128(i128, metadata, metadata)
