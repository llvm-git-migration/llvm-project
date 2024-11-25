// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @emit_vertex() "None" {
    // CHECK: spirv.EmitVertex
    spirv.EmitVertex
    spirv.Return
  }
  spirv.func @end_primitive() "None" {
    // CHECK: spirv.EndPrimitive
    spirv.EndPrimitive
    spirv.Return
  }
}
