// RUN: mlir-opt %s -arm-sme-outer-product-widening -cse -split-input-file | FileCheck %s

// CHECK-LABEL: @outerproduct_add_widening_2way_f16f16f32
// CHECK-SAME:    %[[A0:.*]]: vector<[4]xf16>, %[[B0:.*]]: vector<[4]xf16>, %[[A1:.*]]: vector<[4]xf16>, %[[B1:.*]]: vector<[4]xf16>,
// CHECK-SAME:    %[[A0_MASK:.*]]: vector<[4]xi1>, %[[B0_MASK:.*]]: vector<[4]xi1>, %[[A1_MASK:.*]]: vector<[4]xi1>, %[[B1_MASK:.*]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<[4]x[4]xf32>
// CHECK-DAG: %[[VEC_UNDEF:.*]] = llvm.mlir.undef : vector<[8]xf16>
// CHECK-DAG: %[[A0_INSERT:.*]] = vector.scalable.insert %[[A0]], %[[VEC_UNDEF]][0] : vector<[4]xf16> into vector<[8]xf16>
// CHECK-DAG: %[[B0_INSERT:.*]] = vector.scalable.insert %[[B0]], %[[VEC_UNDEF]][0] : vector<[4]xf16> into vector<[8]xf16>
// CHECK-DAG: %[[A1_INSERT:.*]] = vector.scalable.insert %[[A1]], %[[VEC_UNDEF]][0] : vector<[4]xf16> into vector<[8]xf16>
// CHECK-DAG: %[[B1_INSERT:.*]] = vector.scalable.insert %[[B1]], %[[VEC_UNDEF]][0] : vector<[4]xf16> into vector<[8]xf16>
// CHECK-DAG: %[[LHS:.*]] = "arm_sve.intr.zip1"(%[[A0_INSERT]], %[[A1_INSERT]]) : (vector<[8]xf16>, vector<[8]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[RHS:.*]] = "arm_sve.intr.zip1"(%[[B0_INSERT]], %[[B1_INSERT]]) : (vector<[8]xf16>, vector<[8]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[MASK_UNDEF:.*]] = llvm.mlir.undef : vector<[8]xi1>
// CHECK-DAG: %[[A0_MASK_INSERT:.*]] = vector.scalable.insert %[[A0_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[8]xi1>
// CHECK-DAG: %[[B0_MASK_INSERT:.*]] = vector.scalable.insert %[[B0_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[8]xi1>
// CHECK-DAG: %[[A1_MASK_INSERT:.*]] = vector.scalable.insert %[[A1_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[8]xi1>
// CHECK-DAG: %[[B1_MASK_INSERT:.*]] = vector.scalable.insert %[[B1_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[8]xi1>
// CHECK-DAG: %[[LHS_MASK:.*]] = "arm_sve.intr.zip1"(%[[A0_MASK_INSERT]], %[[A1_MASK_INSERT]]) : (vector<[8]xi1>, vector<[8]xi1>) -> vector<[8]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = "arm_sve.intr.zip1"(%[[B0_MASK_INSERT]], %[[B1_MASK_INSERT]]) : (vector<[8]xi1>, vector<[8]xi1>) -> vector<[8]xi1>
// CHECK-DAG: arm_sme.fmopa_wide_2way %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
func.func @outerproduct_add_widening_2way_f16f16f32(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_f16f16f32
// CHECK: arm_sme.fmops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
func.func @outerproduct_sub_widening_2way_f16f16f32(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_bf16bf16f32
// CHECK: arm_sme.fmopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
func.func @outerproduct_add_widening_2way_bf16bf16f32(
    %a0 : vector<[4]xbf16>, %b0 : vector<[4]xbf16>,
    %a1 : vector<[4]xbf16>, %b1 : vector<[4]xbf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xbf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xbf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xbf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xbf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_bf16bf16f32
// CHECK: arm_sme.fmops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
func.func @outerproduct_sub_widening_2way_bf16bf16f32(
    %a0 : vector<[4]xbf16>, %b0 : vector<[4]xbf16>,
    %a1 : vector<[4]xbf16>, %b1 : vector<[4]xbf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xbf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xbf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xbf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xbf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_signed_i16i16i32
// CHECK: arm_sme.smopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_2way_signed_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_signed_i16i16i32
// CHECK: arm_sme.smops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_2way_signed_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_unsigned_i16i16i32
// CHECK: arm_sme.umopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_2way_unsigned_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extui %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_unsigned_i16i16i32
// CHECK: arm_sme.umops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_2way_unsigned_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extui %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_i8i8i32
// CHECK-SAME:    %[[A0:.*]]: vector<[4]xi8>, %[[B0:.*]]: vector<[4]xi8>, %[[A1:.*]]: vector<[4]xi8>, %[[B1:.*]]: vector<[4]xi8>, %[[A2:.*]]: vector<[4]xi8>, %[[B2:.*]]: vector<[4]xi8>, %[[A3:.*]]: vector<[4]xi8>, %[[B3:.*]]: vector<[4]xi8>,
// CHECK-SAME:    %[[A0_MASK:.*]]: vector<[4]xi1>, %[[B0_MASK:.*]]: vector<[4]xi1>, %[[A1_MASK:.*]]: vector<[4]xi1>, %[[B1_MASK:.*]]: vector<[4]xi1>, %[[A2_MASK:.*]]: vector<[4]xi1>, %[[B2_MASK:.*]]: vector<[4]xi1>, %[[A3_MASK:.*]]: vector<[4]xi1>, %[[B3_MASK:.*]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0> : vector<[4]x[4]xi32>
// CHECK-DAG: %[[VEC_UNDEF:.*]] = llvm.mlir.undef : vector<[16]xi8>
// CHECK-DAG: %[[A0_INSERT:.*]] = vector.scalable.insert %[[A0]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[B0_INSERT:.*]] = vector.scalable.insert %[[B0]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[A1_INSERT:.*]] = vector.scalable.insert %[[A1]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[B1_INSERT:.*]] = vector.scalable.insert %[[B1]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[A2_INSERT:.*]] = vector.scalable.insert %[[A2]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[B2_INSERT:.*]] = vector.scalable.insert %[[B2]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[A3_INSERT:.*]] = vector.scalable.insert %[[A3]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[B3_INSERT:.*]] = vector.scalable.insert %[[B3]], %[[VEC_UNDEF]][0] : vector<[4]xi8> into vector<[16]xi8>
// CHECK-DAG: %[[LHS0:.*]] = "arm_sve.intr.zip1"(%[[A0_INSERT]], %[[A2_INSERT]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[LHS1:.*]] = "arm_sve.intr.zip1"(%[[A1_INSERT]], %[[A3_INSERT]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[RHS0:.*]] = "arm_sve.intr.zip1"(%[[B0_INSERT]], %[[B2_INSERT]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[RHS1:.*]] = "arm_sve.intr.zip1"(%[[B1_INSERT]], %[[B3_INSERT]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[LHS:.*]] = "arm_sve.intr.zip1"(%[[LHS0]], %[[LHS1]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[RHS:.*]] = "arm_sve.intr.zip1"(%[[RHS0]], %[[RHS1]]) : (vector<[16]xi8>, vector<[16]xi8>) -> vector<[16]xi8>
// CHECK-DAG: %[[MASK_UNDEF:.*]] = llvm.mlir.undef : vector<[16]xi1>
// CHECK-DAG: %[[A0_MASK_INSERT:.*]] = vector.scalable.insert %[[A0_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[B0_MASK_INSERT:.*]] = vector.scalable.insert %[[B0_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[A1_MASK_INSERT:.*]] = vector.scalable.insert %[[A1_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[B1_MASK_INSERT:.*]] = vector.scalable.insert %[[B1_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[A2_MASK_INSERT:.*]] = vector.scalable.insert %[[A2_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[B2_MASK_INSERT:.*]] = vector.scalable.insert %[[B2_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[A3_MASK_INSERT:.*]] = vector.scalable.insert %[[A3_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[B3_MASK_INSERT:.*]] = vector.scalable.insert %[[B3_MASK]], %[[MASK_UNDEF]][0] : vector<[4]xi1> into vector<[16]xi1>
// CHECK-DAG: %[[LHS0_MASK:.*]] = "arm_sve.intr.zip1"(%[[A0_MASK_INSERT]], %[[A2_MASK_INSERT]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: %[[LHS1_MASK:.*]] = "arm_sve.intr.zip1"(%[[A1_MASK_INSERT]], %[[A3_MASK_INSERT]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: %[[RHS0_MASK:.*]] = "arm_sve.intr.zip1"(%[[B0_MASK_INSERT]], %[[B2_MASK_INSERT]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: %[[RHS1_MASK:.*]] = "arm_sve.intr.zip1"(%[[B1_MASK_INSERT]], %[[B3_MASK_INSERT]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: %[[LHS_MASK:.*]] = "arm_sve.intr.zip1"(%[[LHS0_MASK]], %[[LHS1_MASK]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = "arm_sve.intr.zip1"(%[[RHS0_MASK]], %[[RHS1_MASK]]) : (vector<[16]xi1>, vector<[16]xi1>) -> vector<[16]xi1>
// CHECK-DAG: arm_sme.smopa_wide_4way %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_i8i8i32
// CHECK: arm_sme.smops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_i16i16i64
// CHECK: arm_sme.smopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_i16i16i64
// CHECK: arm_sme.smops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_i8i8i32
// CHECK: arm_sme.umopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_i8i8i32
// CHECK: arm_sme.umops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_i16i16i64
// CHECK: arm_sme.umopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_i16i16i64
// CHECK: arm_sme.umops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_by_unsigned_i8i8i32
// CHECK: arm_sme.sumopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_signed_by_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_by_unsigned_i8i8i32
// CHECK: arm_sme.sumops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_signed_by_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_by_unsigned_i16i16i64
// CHECK: arm_sme.sumopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_signed_by_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_by_unsigned_i16i16i64
// CHECK: arm_sme.sumops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_signed_by_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_by_signed_i8i8i32
// CHECK: arm_sme.usmopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_unsigned_by_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_by_signed_i8i8i32
// CHECK: arm_sme.usmops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_unsigned_by_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_by_signed_i16i16i64
// CHECK: arm_sme.usmopa_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_unsigned_by_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_by_signed_i16i16i64
// CHECK: arm_sme.usmops_wide_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_unsigned_by_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}
