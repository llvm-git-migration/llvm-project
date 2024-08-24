// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXCHECK

// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPVCHECK

RWBuffer<float> Buf;

// CHECK: define linkonce_odr noundef ptr @"??0?$RWBuffer@M@hlsl@@QAA@XZ"
// CHECK-NEXT: entry:

// DXCHECK: %[[HandleRes:[0-9]+]] = call ptr @llvm.dx.create.handle(i8 1)
// SPVCHECK: %[[HandleRes:[0-9]+]] = call ptr @llvm.spv.create.handle(i8 1)
// CHECK: store ptr %[[HandleRes]], ptr %h, align 4
