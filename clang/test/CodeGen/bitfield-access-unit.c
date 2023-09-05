// Check arches with 32bit ints. (Not you, AVR & MSP430)

// Configs that have cheap unaligned access

// 64-bit Little Endian
// RUN: %clang_cc1 -triple=aarch64-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE,CHECK-64,LAYOUT-64-DWN,PLACE-64-DWN-LE %s
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE,CHECK-64,LAYOUT-64,LAYOUT-64-FLEX,PLACE-64-LE,PLACE-64-FLEX-LE %s
// RUN: %clang_cc1 -triple=loongarch64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE,CHECK-64,LAYOUT-64,LAYOUT-64-FLEX,PLACE-64-LE,PLACE-64-FLEX-LE %s
// RUN: %clang_cc1 -triple=ve-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE %s
// RUN: %clang_cc1 -triple=wasm64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX64-LE,CHECK-64,LAYOUT-64,LAYOUT-64-FLEX,PLACE-64-LE,PLACE-64-FLEX-LE %s

// 64-bit Big Endian
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-BE,PLACE-FLEX-BE,PLACE-FLEX64-BE,CHECK-64,LAYOUT-64,LAYOUT-64-FLEX,PLACE-64-BE,PLACE-64-FLEX-BE %s
// RUN: %clang_cc1 -triple=systemz %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX64,PLACE-BE,PLACE-FLEX-BE,PLACE-FLEX64-BE,CHECK-64,LAYOUT-64,LAYOUT-64-FLEX,PLACE-64-BE,PLACE-64-FLEX-BE %s

// 32-bit Little Endian
// RUN: %clang_cc1 -triple=arm-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN32,LAYOUT-DWN32-FLEX,PLACE-DWN32-LE,PLACE-DWN32-FLEX-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX32,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX32-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX32,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX32-LE %s
// RUN: %clang_cc1 -triple=powerpcle-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX32,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX32-LE %s
// RUN: %clang_cc1 -triple=wasm32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX32,PLACE-LE,PLACE-FLEX-LE,PLACE-FLEX32-LE %s

// 32-bit Big Endian
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-FLEX,LAYOUT-FLEX32,PLACE-BE,PLACE-FLEX-BE,PLACE-FLEX32-BE %s

// Configs that have expensive unaligned access
// 64-bit Little Endian
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -target-feature +strict-align -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE,CHECK-64,LAYOUT-64,LAYOUT-64-STRICT,PLACE-64-LE,PLACE-64-STRICT-LE %s
// RUN: %clang_cc1 -triple=amdgcn-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE,CHECK-64,LAYOUT-64,LAYOUT-64-STRICT,PLACE-64-LE,PLACE-64-STRICT-LE %s
// RUN: %clang_cc1 -triple=loongarch64-elf -target-feature -ual %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE,CHECK-64,LAYOUT-64,LAYOUT-64-STRICT,PLACE-64-LE,PLACE-64-STRICT-LE %s
// RUN: %clang_cc1 -triple=riscv64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE,CHECK-64,LAYOUT-64,LAYOUT-64-STRICT,PLACE-64-LE,PLACE-64-STRICT-LE %s

// 64-big Big endian
// RUN: %clang_cc1 -triple=mips64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-BE,PLACE-STRICT-BE,CHECK-64,LAYOUT-64,LAYOUT-64-STRICT,PLACE-64-BE,PLACE-64-STRICT-BE %s

// 32-bit Little Endian
// RUN: %clang_cc1 -triple=arc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=arm-apple-darwin %s -target-feature +strict-align -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN32,LAYOUT-DWN32-STRICT,PLACE-DWN32-LE,PLACE-DWN32-STRICT-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -target-feature +strict-align -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=bpf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=csky %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=loongarch32-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=nvptx-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=riscv32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=spir-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s
// RUN: %clang_cc1 -triple=xcore-none-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-LE,PLACE-STRICT-LE %s

// 32-bit Big Endian
// RUN: %clang_cc1 -triple=lanai-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-BE,PLACE-STRICT-BE %s
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-BE,PLACE-STRICT-BE %s
// RUN: %clang_cc1 -triple=sparc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-BE,PLACE-STRICT-BE %s
// RUN: %clang_cc1 -triple=tce-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT,LAYOUT-STRICT,PLACE-BE,PLACE-STRICT-BE %s

// Both le64-elf and m68-elf are strict alignment ISAs with 4-byte aligned
// 64-bit or 2-byte aligned 32-bit integer types. This more compex to describe here.

// If unaligned access is expensive don't stick these together.
struct A {
  char a : 7;
  char b : 7;
} a;
// CHECK-LABEL: LLVMType:%struct.A =
// LAYOUT-FLEX-SAME: type { i8, i8 }
// LAYOUT-STRICT-SAME: type { i8, i8 }
// LAYOUT-DWN32-SAME: type { i16 }
// CHECK: BitFields:[
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1

// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-NEXT: ]>

// But do here.
struct __attribute__((aligned(2))) B {
  char a : 7;
  char b : 7;
} b;
// CHECK-LABEL: LLVMType:%struct.B =
// LAYOUT-SAME: type { i8, i8 }
// LAYOUT-DWN32-SAME: type { i16 }
// CHECK: BitFields:[
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-NEXT: ]>

// Not here -- poor alignment within struct
struct C {
  int f1;
  char f2;
  char a : 7;
  char b : 7;
} c;
// CHECK-LABEL: LLVMType:%struct.C =
// LAYOUT-FLEX-SAME: type { i32, i8, i8, i8 }
// LAYOUT-STRICT-SAME: type { i32, i8, i8, i8 }
// LAYOUT-DWN32-SAME: type <{ i32, i8, i16, i8 }>
// CHECK: BitFields:[
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// CHECK-NEXT: ]>

// Not here, we're packed
struct __attribute__((packed)) D {
  int f1;
  int a : 8;
  int b : 8;
  char _;
} d;
// CHECK-LABEL: LLVMType:%struct.D =
// LAYOUT-FLEX-SAME: type <{ i32, i16, i8 }>
// LAYOUT-STRICT-SAME: type <{ i32, i16, i8 }>
// LAYOUT-DWN32-FLEX-SAME: type <{ i32, i16, i8 }>
// LAYOUT-DWN32-STRICT-SAME: type <{ i32, i16, i8 }>
// CHECK: BitFields:[
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-FLEX-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-FLEX-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4

// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4

// PLACE-DWN32-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-DWN32-FLEX-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4

// PLACE-DWN32-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// PLACE-DWN32-STRICT-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// CHECK-NEXT: ]>

struct E {
  char a : 7;
  short b : 13;
  unsigned c : 12;
} e;
// CHECK-LABEL: LLVMType:%struct.E =
// LAYOUT-FLEX64-SAME: type { i8, i16, i16, [2 x i8] }
// LAYOUT-FLEX32-SAME: type { i8, i16, i16, [2 x i8] }
// LAYOUT-STRICT-SAME: type { i8, i16, i16, [2 x i8] }
// LAYOUT-DWN32-SAME: type { i32 }
// CHECK: BitFields:[

// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4

// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4

// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:13 IsSigned:1 StorageSize:32 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:20 Size:12 IsSigned:0 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>

struct F {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
} f;
// CHECK-LABEL: LLVMType:%struct.F =
// LAYOUT-FLEX64-SAME: type { i8, i16, i16, i8 }
// LAYOUT-FLEX32-SAME: type { i8, i16, i16, i8 }
// LAYOUT-STRICT-SAME: type { i8, i16, i16, i8 }
// LAYOUT-DWN32-SAME: type { [5 x i8] }
// CHECK: BitFields:[
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-FLEX32-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-STRICT-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:13 IsSigned:1 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:20 Size:12 IsSigned:0 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:32 Size:7 IsSigned:1 StorageSize:40 StorageOffset:0
// CHECK-NEXT: ]>

struct G {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
  signed char e;
} g;
// CHECK-LABEL: LLVMType:%struct.G =
// LAYOUT-SAME: type { i8, i16, i16, i8, i8 }
// LAYOUT-DWN32-SAME: type { [5 x i8], i8 }
// CHECK: BitFields:[
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// PLACE-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6

// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:7 Size:13 IsSigned:1 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:20 Size:12 IsSigned:0 StorageSize:40 StorageOffset:0
// PLACE-DWN32-LE-NEXT: <CGBitFieldInfo Offset:32 Size:7 IsSigned:1 StorageSize:40 StorageOffset:0
// CHECK-NEXT: ]>

#if _LP64
struct A64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e : 8;
} a64;
// CHECK-64-LABEL: LLVMType:%struct.A64 =
// LAYOUT-64-SAME: type { i64 }
// LAYOUT-64-DWN-SAME: type { i64 }
// CHECK-64: BitFields:[
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:56 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:48 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:40 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0

// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:56 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK-64-NEXT: ]>

struct B64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e; // not a bitfield
} b64;
// CHECK-64-LABEL: LLVMType:%struct.B64 =
// LAYOUT-64-FLEX-SAME: type { [7 x i8], i8 }
// LAYOUT-64-STRICT-SAME: type { [7 x i8], i8 }
// LAYOUT-64-DWN-SAME: type { [7 x i8], i8 }
// CHECK-64: BitFields:[
// PLACE-64-FLEX-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-BE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-FLEX-BE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0

// PLACE-64-STRICT-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-BE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-STRICT-BE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0

// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK-64-NEXT: ]>

struct C64 {
  int a : 15;
  short b : 8;
  long c : 16;
  int d : 15;
  signed char e : 7;
} c64;
// CHECK-64-LABEL: LLVMType:%struct.C64 =
// LAYOUT-64-SAME: type { i16, [5 x i8], i8 }
// LAYOUT-64-DWN-SAME: type { i16, [5 x i8], i8 }
// CHECK-64: BitFields:[
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:15 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:24 Size:15 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:7
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:15 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:15 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:7

// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:0 Size:15 IsSigned:1 StorageSize:16 StorageOffset:0
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:24 Size:15 IsSigned:1 StorageSize:40 StorageOffset:2
// PLACE-64-DWN-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:7
// CHECK-64-NEXT: ]>

#endif
