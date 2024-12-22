// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test OpAsmOpInterface
//===----------------------------------------------------------------------===//

func.func @result_name_from_op_asm_type_interface() {
  // CHECK-LABEL: @result_name_from_op_asm_type_interface
  // CHECK: %op_asm_type_interface
  %0 = "test.default_result_name"() : () -> !test.op_asm_type_interface
  return
}

// -----

func.func @result_name_pack_from_op_asm_type_interface() {
  // CHECK-LABEL: @result_name_pack_from_op_asm_type_interface
  // CHECK: %op_asm_type_interface{{.*}}, %op_asm_type_interface{{.*}}
  // CHECK-NOT: :2
  %0:2 = "test.default_result_name_packing"() : () -> (!test.op_asm_type_interface, !test.op_asm_type_interface)
  return
}

// -----

func.func @result_name_pack_do_nothing() {
  // CHECK-LABEL: @result_name_pack_do_nothing
  // CHECK: %0:2
  %0:2 = "test.default_result_name_packing"() : () -> (i32, !test.op_asm_type_interface)
  return
}

// -----

func.func @block_argument_name_from_op_asm_type_interface() {
  // CHECK-LABEL: @block_argument_name_from_op_asm_type_interface
  // CHECK: ^bb0(%op_asm_type_interface
  test.default_block_argument_name {
    ^bb0(%arg0: !test.op_asm_type_interface):
      "test.terminator"() : ()->()
  }
  return
}