// Check arches with 32bit ints. (Not you, AVR & MSP430)

// Configs that have cheap unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=loongarch64-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=powerpcle-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=ve-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=wasm32 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=wasm64 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// Big Endian, you weirdos
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s
// RUN: %clang_cc1 -triple=systemz %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s

// Configs that have expensive unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=amdgcn-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=arc-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=bpf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=csky %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=le64-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=loongarch32-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=nvptx-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=riscv32 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=riscv64 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=spir-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=xcore-none-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// Big endian, you're lovely
// RUN: %clang_cc1 -triple=lanai-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=m68k-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=mips64-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=sparc-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=tce-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s

// If unaligned access is expensive don't stick these together.
struct A {
  char a : 7;
  char b : 7;
} a;
// CHECK-LABEL: LLVMType:%struct.A =
// FLEXO-SAME: type { i16 }
// STRICTO-SAME: type { i8, i8 }
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:8 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// CHECK-NEXT: ]>

// But do here.
struct __attribute__((aligned(2))) B {
  char a : 7;
  char b : 7;
} b;
// CHECK-LABEL: LLVMType:%struct.B =
// CHECK-SAME: type { i16 }
// CHECK: BitFields:[
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:8 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK-NEXT: ]>

// Not here -- poor alignment within struct
struct C {
  int f1;
  char f2;
  char a : 7;
  char b : 7;
} c;
// CHECK-LABEL: LLVMType:%struct.C =
// FLEXO-SAME: type <{ i32, i8, i16, i8 }>
// STRICTO-SAME: type { i32, i8, i8, i8 }
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:8 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:16 StorageOffset:5
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// CHECK-NEXT: ]>

// Not here, we're packed
struct __attribute__((packed)) D {
  int f1;
  int a : 8;
  int b : 8;
  char _;
} d;
// CHECK-LABEL: LLVMType:%struct.D =
// FLEXO-SAME: type <{ i32, i16, i8 }>
// STRICTO-SAME: type <{ i32, i8, i8, i8 }>
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:4
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:4
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:5
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:4
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:5
// CHECK-NEXT: ]>
