// DEFINE: %{entry} = test_outerproduct_i8i8i32
// DEFINE: %{widening_opts} = -arm-sme-outer-product-widening
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za" \
// DEFINE:   -convert-vector-to-arm-sme %{widening_opts} \
// DEFINE:   -convert-arm-sme-to-scf -allocate-arm-sme-tiles \
// DEFINE:   -convert-arm-sme-to-llvm -cse -canonicalize \
// DEFINE:   -test-lower-to-llvm -o %t
// DEFINE: %{run} = %mcr_aarch64_cmd %t \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry} -entry-point-result=void \
// DEFINE:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib

// RUN: %{compile}

// RUN: %{run} | FileCheck %s

// REDEFINE: %{entry} = test_masked_outerproduct_i8i8i32
// RUN: %{run} | FileCheck %s --check-prefix=WITH-MASK

// NOTE: QEMU gives incorrect result for SME SMOPA 4-way outer product
// instruction (version <= 8.2.0, latest version at time of writing), see:
// https://gitlab.com/qemu-project/qemu/-/issues/2083
// This test is expected to fail until a fixed version of QEMU can be used.

// FIXME: Remove the 'XFAIL' below once a fixed QEMU version is available
// (and installed on CI buildbot).
// XFAIL: *

// NOTE: there is no non-widening variant for these types and this test can't
// currently be lowered without the widening pass, therefore we can't check if
// the result is the same without widening pass like
// 'test-outerproduct-f16f16f32.mlir' does.

func.func @test_outerproduct_i8i8i32() {
  %undef = llvm.mlir.undef : vector<[4]xi8>

  %a0_data = arith.constant dense<[0, 4, 8, 12]> : vector<4xi8>
  %a1_data = arith.constant dense<[1, 5, 9, 13]> : vector<4xi8>
  %a2_data = arith.constant dense<[2, 6, 10, 14]> : vector<4xi8>
  %a3_data = arith.constant dense<[3, 7, 11, 15]> : vector<4xi8>

  %b0_data = arith.constant dense<[16, 20, 24, 28]> : vector<4xi8>
  %b1_data = arith.constant dense<[17, 21, 25, 29]> : vector<4xi8>
  %b2_data = arith.constant dense<[18, 22, 26, 30]> : vector<4xi8>
  %b3_data = arith.constant dense<[19, 23, 27, 31]> : vector<4xi8>

  %a0 = vector.scalable.insert %a0_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b0 = vector.scalable.insert %b0_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a1 = vector.scalable.insert %a1_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b1 = vector.scalable.insert %b1_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a2 = vector.scalable.insert %a2_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b2 = vector.scalable.insert %b2_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a3 = vector.scalable.insert %a3_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b3 = vector.scalable.insert %b3_data, %undef[0] : vector<4xi8> into vector<[4]xi8>

  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>
  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>
  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = vector.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = vector.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xi32>, vector<[4]xi32>
  %2 = vector.outerproduct %a2_ext, %b2_ext, %1 : vector<[4]xi32>, vector<[4]xi32>
  %3 = vector.outerproduct %a3_ext, %b3_ext, %2 : vector<[4]xi32>, vector<[4]xi32>

  // CHECK:      ( 110,  134,  158,  182
  // CHECK-NEXT: ( 390,  478,  566,  654
  // CHECK-NEXT: ( 670,  822,  974, 1126
  // CHECK-NEXT: ( 950, 1166, 1382, 1598
  vector.print %3 : vector<[4]x[4]xi32>

  return
}

func.func @test_masked_outerproduct_i8i8i32() {
  %undef = llvm.mlir.undef : vector<[4]xi8>

  %a0_data = arith.constant dense<[0, 4, 8, 12]> : vector<4xi8>
  %a1_data = arith.constant dense<[1, 5, 9, 13]> : vector<4xi8>
  %a2_data = arith.constant dense<[2, 6, 10, 14]> : vector<4xi8>
  %a3_data = arith.constant dense<[3, 7, 11, 15]> : vector<4xi8>

  %b0_data = arith.constant dense<[16, 20, 24, 28]> : vector<4xi8>
  %b1_data = arith.constant dense<[17, 21, 25, 29]> : vector<4xi8>
  %b2_data = arith.constant dense<[18, 22, 26, 30]> : vector<4xi8>
  %b3_data = arith.constant dense<[19, 23, 27, 31]> : vector<4xi8>

  %a0 = vector.scalable.insert %a0_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b0 = vector.scalable.insert %b0_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a1 = vector.scalable.insert %a1_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b1 = vector.scalable.insert %b1_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a2 = vector.scalable.insert %a2_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b2 = vector.scalable.insert %b2_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %a3 = vector.scalable.insert %a3_data, %undef[0] : vector<4xi8> into vector<[4]xi8>
  %b3 = vector.scalable.insert %b3_data, %undef[0] : vector<4xi8> into vector<[4]xi8>

  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>
  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>
  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %mask0 = vector.create_mask %c1, %c1 : vector<[4]x[4]xi1>
  %mask1 = vector.create_mask %c1, %c2 : vector<[4]x[4]xi1>
  %mask2 = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %mask3 = vector.create_mask %c3, %c4 : vector<[4]x[4]xi1>

  %acc = arith.constant dense<2> : vector<[4]x[4]xi32>
  %0 = vector.mask %mask0 {
    vector.outerproduct %a0_ext, %b0_ext, %acc : vector<[4]xi32>, vector<[4]xi32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xi32>
  %1 = vector.mask %mask1 {
    vector.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xi32>, vector<[4]xi32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xi32>
  %2 = vector.mask %mask2 {
    vector.outerproduct %a2_ext, %b2_ext, %1 : vector<[4]xi32>, vector<[4]xi32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xi32>
  %3 = vector.mask %mask3 {
    vector.outerproduct %a3_ext, %b3_ext, %2 : vector<[4]xi32>, vector<[4]xi32>
  } : vector<[4]x[4]xi1> -> vector<[4]x[4]xi32>

  // WITH-MASK:      ( 112, 136, 135,  95
  // WITH-MASK-NEXT: ( 243, 295, 347, 219
  // WITH-MASK-NEXT: ( 211, 255, 299, 343
  // WITH-MASK-NEXT: (   2,   2,   2,   2
  vector.print %3 : vector<[4]x[4]xi32>

  return
}
