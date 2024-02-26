// Check arches with 32bit ints. (Not you, AVR & MSP430)

// Configs that have cheap unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=loongarch64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=powerpcle-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=ve-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=wasm32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=wasm64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,FLEXO,FLEXO-LE %s
// Big Endian
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s
// RUN: %clang_cc1 -triple=systemz %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,FLEXO,FLEXO-BE %s

// Configs that have expensive unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -target-feature +strict-align -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=amdgcn-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=arc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -target-feature +strict-align -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=bpf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=csky %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=loongarch64-elf -target-feature -ual %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=loongarch32-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=nvptx-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=riscv32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=riscv64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=spir-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// RUN: %clang_cc1 -triple=xcore-none-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE,STRICTO,STRICTO-LE %s
// Big endian
// RUN: %clang_cc1 -triple=lanai-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=mips64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=sparc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s
// RUN: %clang_cc1 -triple=tce-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE,STRICTO,STRICTO-BE %s

// Both le64-elf and m68-elf are strict alignment ISAs with 4-byte aligned
// 64-bit or 2-byte aligned 32-bit integer types. This more compex to describe here.

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

struct E {
  char a : 7;
  short b : 13;
  unsigned c : 12;
} e;
// CHECK-LABEL: LLVMType:%struct.E =
// FLEXO-SAME: type <{ i8, i8, i32, [2 x i8] }>
// STRICTO-SAME: type { i8, i16, i16, [2 x i8] }
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:16 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:19 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// CHECK-NEXT: ]>

struct F {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
} f;
// CHECK-LABEL: LLVMType:%struct.F =
// FLEXO-SAME: type <{ i8, i8, i32, i8, i8 }>
// STRICTO-SAME: type { i8, i16, i32 }
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:16 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:19 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:32 StorageOffset:4
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:16 Size:7 IsSigned:1 StorageSize:32 StorageOffset:4
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:20 Size:12 IsSigned:0 StorageSize:32 StorageOffset:4
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:9 Size:7 IsSigned:1 StorageSize:32 StorageOffset:4
// CHECK-NEXT: ]>

struct G {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
  signed char e;
} g;
// CHECK-LABEL: LLVMType:%struct.G =
// FLEXO-SAME: type <{ i8, i8, i32, i8, i8 }>
// STRICTO-SAME: type { i8, i16, i16, i8, i8 }
// CHECK: BitFields:[
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:16 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// FLEXO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:19 Size:13 IsSigned:1 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:32 StorageOffset:2
// FLEXO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// STRICTO-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:16 StorageOffset:2
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// STRICTO-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// CHECK-NEXT: ]>

#if _LP64
struct A64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e : 8;
} a64;
// CHECK64-LABEL: LLVMType:%struct.A64 =
// CHECK64-SAME: type { i64 }
// CHECK64: BitFields:[
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:56 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:48 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:40 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// CHECK64-NEXT: ]>

struct B64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e; // not a bitfield
} b64;
// CHECK64-LABEL: LLVMType:%struct.B64 =
// CHECK64-SAME: type { [7 x i8], i8 }
// CHECK64: BitFields:[
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:56 StorageOffset:0
// CHECK64-NEXT: ]>

struct C64 {
  int a : 15;
  short b : 8;
  long c : 16;
  int d : 15;
  signed char e : 7;
} c64;
// CHECK64-LABEL: LLVMType:%struct.C64 =
// CHECK64-SAME: type {  i16, [5 x i8], i8 }
// CHECK64: BitFields:[
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:15 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:24 Size:15 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:7
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:15 IsSigned:1 StorageSize:16 StorageOffset:0
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:15 IsSigned:1 StorageSize:40 StorageOffset:2
// CHECK64-BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:7
// CHECK64-NEXT: ]>

#endif
