// Tests for bitfield access with zero-length bitfield padding

// Configs that have cheap unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=arm-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN32,PLACE-DWN32-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s

// Big Endian
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-BE %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-BE %s

// Configs that have expensive unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=le64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-LE %s

// Big endian
// RUN: %clang_cc1 -triple=m68k-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-BE %s
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-T,PLACE-T-BE %s

// And now a few with -fno-bitfield-type-align. Precisely how this behaves is
// ABI-dependent.
// Cheap unaligned
// RUN: %clang_cc1 -triple=aarch64-apple-darwin -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-NT,PLACE-NT-LE %s
// RUN: %clang_cc1 -triple=aarch64-linux-gnu -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN64-T,PLACE-T-LE %s
// RUN: %clang_cc1 -triple=arm-apple-darwin -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN32,PLACE-DWN32-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-NT,PLACE-NT-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-NT,PLACE-NT-LE %s
// RUN: %clang_cc1 -triple=powerpc-linux-gnu -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-NT,PLACE-NT-BE %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-NT,PLACE-NT-BE %s

// Expensive unaligned
// RUN: %clang_cc1 -triple=hexagon-elf -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-STRICT-NT,PLACE-STRICT-NT-LE %s
// RUN: %clang_cc1 -triple=mips-elf -fno-bitfield-type-align %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-STRICT-NT,PLACE-STRICT-NT-BE %s

struct P1 {
  unsigned a :8;
  char :0;
  unsigned b :8;
} p1;
// CHECK-LABEL: LLVMType:%struct.P1 =
// LAYOUT-T-SAME: type { i8, i8, [2 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:1
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:1

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

// This will often be align(1) with -fno-bitfield-type-align
struct P2 {
  unsigned a :8;
  char :0;
  short :0;
  unsigned b :8;
} p2;
// CHECK-LABEL: LLVMType:%struct.P2 =
// LAYOUT-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-DWN64-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P3 {
  unsigned a :8;
  char :0;
  short :0;
  unsigned :0;
  unsigned b :8;
} p3;
// CHECK-LABEL: LLVMType:%struct.P3 =
// LAYOUT-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P4 {
  unsigned a :8;
  short :0;
  unsigned :0;
  unsigned b :8;
} p4;
// CHECK-LABEL: LLVMType:%struct.P4 =
// LAYOUT-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P5 {
  unsigned a :8;
  unsigned :0;
  unsigned b :8;
} p5;
// CHECK-LABEL: LLVMType:%struct.P5 =
// LAYOUT-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P6 {
  unsigned a :8;
  unsigned :0;
  short :0;
  char :0;
  unsigned b :8;
} p6;
// CHECK-LABEL: LLVMType:%struct.P6 =
// LAYOUT-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P7 {
  unsigned a : 8;
  short : 0;
  unsigned char b : 8;
} p7;
// CHECK-LABEL: LLVMType:%struct.P7 =
// LAYOUT-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-DWN64-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

// And with forced alignment for !useZeroLengthBitfieldAlignment machines (eg
// hexagon)
struct __attribute__ ((aligned (2))) P7_align {
  unsigned a : 8;
  short : 0;
  unsigned char b : 8;
} p7_align;
// CHECK-LABEL: LLVMType:%struct.P7_align =
// LAYOUT-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-DWN64-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:2

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P8 {
  unsigned a : 7;
  short : 0;
  unsigned char b : 7;
} p8;
// CHECK-LABEL: LLVMType:%struct.P8 =
// LAYOUT-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-DWN64-T-SAME: type { i8, i8, i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:2
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:0 StorageSize:8 StorageOffset:2

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:2 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:2 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct P9 {
  unsigned a : 7;
  char : 0;
  unsigned short b : 7;
} p9;
// CHECK-LABEL: LLVMType:%struct.P9 =
// LAYOUT-T-SAME: type { i8, i8, [2 x i8] }
// LAYOUT-DWN64-T-SAME: type { i8, i8 }
// LAYOUT-NT-SAME: type { i16 }
// LAYOUT-STRICT-NT-SAME: type { i16 }
// LAYOUT-DWN32-SAME: type { i8, [3 x i8], i8, [3 x i8] }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:1
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:0 StorageSize:8 StorageOffset:1

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:2 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:9 Size:0 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:2 Size:7 IsSigned:0 StorageSize:16 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:8 StorageOffset:4
// CHECK-NEXT: ]>

struct __attribute__((aligned(4))) P10 {
  unsigned a : 7;
  unsigned short b : 7;
  unsigned c : 7;
  char : 0;
} p10;
// CHECK-LABEL: LLVMType:%struct.P10 =
// LAYOUT-T-SAME: type { i24 }
// LAYOUT-DWN64-T-SAME: type { i24 }
// LAYOUT-NT-SAME: type { i24 }
// LAYOUT-STRICT-NT-SAME: type { i24 }
// LAYOUT-DWN32-SAME: type { i24 }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:14 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:11 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:14 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:21 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:11 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:11 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:14 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:21 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:11 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:11 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:14 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>

struct __attribute__((aligned(4))) P11 {
  unsigned a : 7;
  unsigned short b : 7;
  unsigned c : 10;
  char : 0; // at a char boundary
} p11;
// CHECK-LABEL: LLVMType:%struct.P11 =
// LAYOUT-T-SAME: type { i24 }
// LAYOUT-DWN64-T-SAME: type { i24 }
// LAYOUT-NT-SAME: type { i24 }
// LAYOUT-STRICT-NT-SAME: type { i24 }
// LAYOUT-DWN32-SAME: type { i24 }
// CHECK: BitFields:[
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-LE-NEXT: <CGBitFieldInfo Offset:14 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-T-BE-NEXT: <CGBitFieldInfo Offset:8 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0

// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:14 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-LE-NEXT: <CGBitFieldInfo Offset:24 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0

// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:14 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-LE-NEXT: <CGBitFieldInfo Offset:24 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:25 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:18 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-STRICT-NT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:0 IsSigned:1 StorageSize:32 StorageOffset:0

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:0 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:14 Size:10 IsSigned:0 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>
