; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for any are generated for float and half.


target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"


; CHECK:icmp ne i1 %{{.*}}, false
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_bool(i1 noundef %p0) #0 {
entry:
  %p0.addr = alloca i8, align 1
  %frombool = zext i1 %p0 to i8
  store i8 %frombool, ptr %p0.addr, align 1
  %0 = load i8, ptr %p0.addr, align 1
  %tobool = trunc i8 %0 to i1
  %dx.any = call i1 @llvm.dx.any.i1(i1 %tobool)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.i1(i1) #1

; CHECK:icmp ne i64 %{{.*}}, 0
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_int64_t(i64 noundef %p0) #0 {
entry:
  %p0.addr = alloca i64, align 8
  store i64 %p0, ptr %p0.addr, align 8
  %0 = load i64, ptr %p0.addr, align 8
  %dx.any = call i1 @llvm.dx.any.i64(i64 %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.i64(i64) #1

; CHECK:icmp ne i32 %{{.*}}, 0
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_int(i32 noundef %p0) #0 {
entry:
  %p0.addr = alloca i32, align 4
  store i32 %p0, ptr %p0.addr, align 4
  %0 = load i32, ptr %p0.addr, align 4
  %dx.any = call i1 @llvm.dx.any.i32(i32 %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.i32(i32) #1

; CHECK:icmp ne i16 %{{.*}}, 0
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_int16_t(i16 noundef %p0) #0 {
entry:
  %p0.addr = alloca i16, align 2
  store i16 %p0, ptr %p0.addr, align 2
  %0 = load i16, ptr %p0.addr, align 2
  %dx.any = call i1 @llvm.dx.any.i16(i16 %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.i16(i16) #1

; CHECK:fcmp une double %{{.*}}, 0.000000e+00
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_double(double noundef %p0) #0 {
entry:
  %p0.addr = alloca double, align 8
  store double %p0, ptr %p0.addr, align 8
  %0 = load double, ptr %p0.addr, align 8
  %dx.any = call i1 @llvm.dx.any.f64(double %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.f64(double) #1

; CHECK:fcmp une float %{{.*}}, 0.000000e+00
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_float(float noundef %p0) #0 {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  %dx.any = call i1 @llvm.dx.any.f32(float %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.f32(float) #1

; CHECK:fcmp une half %{{.*}}, 0xH0000
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_half(half noundef %p0) #0 {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %dx.any = call i1 @llvm.dx.any.f16(half %0)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.f16(half) #1

; CHECK:icmp ne <4 x i1> %extractvec, zeroinitialize
; CHECK:extractelement <4 x i1> %{{.*}}, i64 0
; CHECK:extractelement <4 x i1> %{{.*}}, i64 1
; CHECK:or i1  %{{.*}}, %{{.*}}
; CHECK:extractelement <4 x i1> %{{.*}}, i64 2
; CHECK:or i1  %{{.*}}, %{{.*}}
; CHECK:extractelement <4 x i1> %{{.*}}, i64 3
; CHECK:or i1  %{{.*}}, %{{.*}}
; Function Attrs: noinline nounwind optnone
define noundef i1 @any_bool4(<4 x i1> noundef %p0) #0 {
entry:
  %p0.addr = alloca i8, align 1
  %insertvec = shufflevector <4 x i1> %p0, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %0 = bitcast <8 x i1> %insertvec to i8
  store i8 %0, ptr %p0.addr, align 1
  %load_bits = load i8, ptr %p0.addr, align 1
  %1 = bitcast i8 %load_bits to <8 x i1>
  %extractvec = shufflevector <8 x i1> %1, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %dx.any = call i1 @llvm.dx.any.v4i1(<4 x i1> %extractvec)
  ret i1 %dx.any
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i1 @llvm.dx.any.v4i1(<4 x i1>) #1
