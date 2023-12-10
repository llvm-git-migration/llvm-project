// Check we use tail padding if it is known to be safe

// Configs that have cheap unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=powerpcle-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// Big Endian, you weirdos
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE %s

// Configs that have expensive unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=amdgcn-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=riscv32 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// RUN: %clang_cc1 -triple=riscv64 %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-LE %s
// Big endian, you're lovely
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE %s
// RUN: %clang_cc1 -triple=mips64-elf %s -emit-llvm -o %t -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,CHECK-BE %s

// Can use tail padding
struct Pod {
  int a : 16;
  int b : 8;
} P;
// CHECK-LABEL: LLVMType:%struct.Pod =
// CHECK-SAME: type { i24 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.Pod =
// CHECK: BitFields:[
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>

// No tail padding
struct __attribute__((packed)) PPod {
  int a : 16;
  int b : 8;
} PP;
// CHECK-LABEL: LLVMType:%struct.PPod =
// CHECK-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.PPod =
// CHECK: BitFields:[
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>

// Cannot use tail padding
struct NonPod {
  ~NonPod();
  int a : 16;
  int b : 8;
} NP;
// CHECK-LABEL: LLVMType:%struct.NonPod =
// CHECK-SAME: type { [3 x i8], i8 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.NonPod.base = type { [3 x i8] }
// CHECK: BitFields:[
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>

// No tail padding
struct __attribute__((packed)) PNonPod {
  ~PNonPod();
  int a : 16;
  int b : 8;
} PNP;
// CHECK-LABEL: LLVMType:%struct.PNonPod =
// CHECK-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.PNonPod =
// CHECK: BitFields:[
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-LE-NEXT: <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>
