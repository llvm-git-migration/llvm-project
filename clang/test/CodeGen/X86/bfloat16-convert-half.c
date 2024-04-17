// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-O0-optnone -emit-llvm \
// RUN:   %s -o - | opt -S -passes=mem2reg | FileCheck %s

// CHECK-LABEL: define dso_local half @test_convert_from_bf16_to_fp16(
// CHECK-SAME: bfloat noundef [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext bfloat [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to half
// CHECK-NEXT:    ret half [[FPTRUNC]]
//
_Float16 test_convert_from_bf16_to_fp16(__bf16 a) {
    return (_Float16)a;
}

// CHECK-LABEL: define dso_local bfloat @test_convert_from_fp16_to_bf16(
// CHECK-SAME: half noundef [[A:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext half [[A]] to float
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc float [[FPEXT]] to bfloat
// CHECK-NEXT:    ret bfloat [[FPTRUNC]]
//
__bf16 test_convert_from_fp16_to_bf16(_Float16 a) {
    return (__bf16)a;
}

typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));

typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));
typedef __bf16 bfloat4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define dso_local i32 @test_cast_from_half2_to_bfloat2(
// CHECK-SAME: i32 noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <2 x bfloat>, align 4
// CHECK-NEXT:    [[IN:%.*]] = alloca <2 x half>, align 4
// CHECK-NEXT:    store i32 [[IN_COERCE]], ptr [[IN]], align 4
// CHECK-NEXT:    [[IN1:%.*]] = load <2 x half>, ptr [[IN]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x half> [[IN1]] to <2 x bfloat>
// CHECK-NEXT:    store <2 x bfloat> [[TMP0]], ptr [[RETVAL]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP1]]
//
bfloat2 test_cast_from_half2_to_bfloat2(half2 in) {
  return (bfloat2)in;
}


// CHECK-LABEL: define dso_local double @test_cast_from_half4_to_bfloat4(
// CHECK-SAME: double noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <4 x bfloat>, align 8
// CHECK-NEXT:    [[IN:%.*]] = alloca <4 x half>, align 8
// CHECK-NEXT:    store double [[IN_COERCE]], ptr [[IN]], align 8
// CHECK-NEXT:    [[IN1:%.*]] = load <4 x half>, ptr [[IN]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x half> [[IN1]] to <4 x bfloat>
// CHECK-NEXT:    store <4 x bfloat> [[TMP0]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret double [[TMP1]]
//
bfloat4 test_cast_from_half4_to_bfloat4(half4 in) {
  return (bfloat4)in;
}

// CHECK-LABEL: define dso_local i32 @test_cast_from_bfloat2_to_half2(
// CHECK-SAME: i32 noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <2 x half>, align 4
// CHECK-NEXT:    [[IN:%.*]] = alloca <2 x bfloat>, align 4
// CHECK-NEXT:    store i32 [[IN_COERCE]], ptr [[IN]], align 4
// CHECK-NEXT:    [[IN1:%.*]] = load <2 x bfloat>, ptr [[IN]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x bfloat> [[IN1]] to <2 x half>
// CHECK-NEXT:    store <2 x half> [[TMP0]], ptr [[RETVAL]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP1]]
//
half2 test_cast_from_bfloat2_to_half2(bfloat2 in) {
  return (half2)in;
}


// CHECK-LABEL: define dso_local double @test_cast_from_bfloat4_to_half4(
// CHECK-SAME: double noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <4 x half>, align 8
// CHECK-NEXT:    [[IN:%.*]] = alloca <4 x bfloat>, align 8
// CHECK-NEXT:    store double [[IN_COERCE]], ptr [[IN]], align 8
// CHECK-NEXT:    [[IN1:%.*]] = load <4 x bfloat>, ptr [[IN]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[IN1]] to <4 x half>
// CHECK-NEXT:    store <4 x half> [[TMP0]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret double [[TMP1]]
//
half4 test_cast_from_bfloat4_to_half4(bfloat4 in) {
  return (half4)in;
}


// CHECK-LABEL: define dso_local i32 @test_convertvector_from_half2_to_bfloat2(
// CHECK-SAME: i32 noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <2 x bfloat>, align 4
// CHECK-NEXT:    [[IN:%.*]] = alloca <2 x half>, align 4
// CHECK-NEXT:    store i32 [[IN_COERCE]], ptr [[IN]], align 4
// CHECK-NEXT:    [[IN1:%.*]] = load <2 x half>, ptr [[IN]], align 4
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <2 x half> [[IN1]] to <2 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <2 x float> [[FPEXT]] to <2 x bfloat>
// CHECK-NEXT:    store <2 x bfloat> [[FPTRUNC]], ptr [[RETVAL]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
bfloat2 test_convertvector_from_half2_to_bfloat2(half2 in) {
  return __builtin_convertvector(in, bfloat2);
}

// CHECK-LABEL: define dso_local i32 @test_convertvector_from_bfloat2_to_half2(
// CHECK-SAME: i32 noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <2 x half>, align 4
// CHECK-NEXT:    [[IN:%.*]] = alloca <2 x bfloat>, align 4
// CHECK-NEXT:    store i32 [[IN_COERCE]], ptr [[IN]], align 4
// CHECK-NEXT:    [[IN1:%.*]] = load <2 x bfloat>, ptr [[IN]], align 4
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <2 x bfloat> [[IN1]] to <2 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <2 x float> [[FPEXT]] to <2 x half>
// CHECK-NEXT:    store <2 x half> [[FPTRUNC]], ptr [[RETVAL]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
half2 test_convertvector_from_bfloat2_to_half2(bfloat2 in) {
  return __builtin_convertvector(in, half2);
}

// CHECK-LABEL: define dso_local double @test_convertvector_from_half4_to_bfloat4(
// CHECK-SAME: double noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <4 x bfloat>, align 8
// CHECK-NEXT:    [[IN:%.*]] = alloca <4 x half>, align 8
// CHECK-NEXT:    store double [[IN_COERCE]], ptr [[IN]], align 8
// CHECK-NEXT:    [[IN1:%.*]] = load <4 x half>, ptr [[IN]], align 8
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x half> [[IN1]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x bfloat>
// CHECK-NEXT:    store <4 x bfloat> [[FPTRUNC]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret double [[TMP0]]
//
bfloat4 test_convertvector_from_half4_to_bfloat4(half4 in) {
  return __builtin_convertvector(in, bfloat4);
}

// CHECK-LABEL: define dso_local double @test_convertvector_from_bfloat4_to_half4(
// CHECK-SAME: double noundef [[IN_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca <4 x half>, align 8
// CHECK-NEXT:    [[IN:%.*]] = alloca <4 x bfloat>, align 8
// CHECK-NEXT:    store double [[IN_COERCE]], ptr [[IN]], align 8
// CHECK-NEXT:    [[IN1:%.*]] = load <4 x bfloat>, ptr [[IN]], align 8
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x bfloat> [[IN1]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x half>
// CHECK-NEXT:    store <4 x half> [[FPTRUNC]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret double [[TMP0]]
//
half4 test_convertvector_from_bfloat4_to_half4(bfloat4 in) {
  return __builtin_convertvector(in, half4);
}
