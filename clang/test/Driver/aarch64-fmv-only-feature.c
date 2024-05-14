// Test that features which are meaningful only for Function Multiversioning are rejected from the command line.

// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+dgh %s 2>&1 | FileCheck %s --check-prefix=DGH
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+ebf16 %s 2>&1 | FileCheck %s --check-prefix=EBF16
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+ls64_accdata %s 2>&1 | FileCheck %s --check-prefix=LS64_ACCDATA
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+ls64_v %s 2>&1 | FileCheck %s --check-prefix=LS64_V
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+memtag2 %s 2>&1 | FileCheck %s --check-prefix=MEMTAG2
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+memtag3 %s 2>&1 | FileCheck %s --check-prefix=MEMTAG3
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+pmull %s 2>&1 | FileCheck %s --check-prefix=PMULL
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+rpres %s 2>&1 | FileCheck %s --check-prefix=RPRES
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+sha1 %s 2>&1 | FileCheck %s --check-prefix=SHA1
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+ssbs2 %s 2>&1 | FileCheck %s --check-prefix=SSBS2
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+sve-bf16 %s 2>&1 | FileCheck %s --check-prefix=SVE_BF16
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+sve-ebf16 %s 2>&1 | FileCheck %s --check-prefix=SVE_EBF16
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+sve-i8mm %s 2>&1 | FileCheck %s --check-prefix=SVE_I8MM
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+sve2-pmull128 %s 2>&1 | FileCheck %s --check-prefix=SVE2_PMULL128
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+dpb %s 2>&1 | FileCheck %s --check-prefix=DPB
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+rcpc2 %s 2>&1 | FileCheck %s --check-prefix=RCPC2
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+flagm2 %s 2>&1 | FileCheck %s --check-prefix=FLAGM2
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+frintts %s 2>&1 | FileCheck %s --check-prefix=FRINTTS
// RUN: not %clang --target=aarch64-linux-gnu -march=armv8-a+dpb2 %s 2>&1 | FileCheck %s --check-prefix=DPB2

// DGH: error: unsupported argument 'armv8-a+dgh' to option '-march='
// EBF16: error: unsupported argument 'armv8-a+ebf16' to option '-march='
// LS64_ACCDATA: error: unsupported argument 'armv8-a+ls64_accdata' to option '-march='
// LS64_V: error: unsupported argument 'armv8-a+ls64_v' to option '-march='
// MEMTAG2: error: unsupported argument 'armv8-a+memtag2' to option '-march='
// MEMTAG3: error: unsupported argument 'armv8-a+memtag3' to option '-march='
// PMULL: error: unsupported argument 'armv8-a+pmull' to option '-march='
// RPRES: error: unsupported argument 'armv8-a+rpres' to option '-march='
// SHA1: error: unsupported argument 'armv8-a+sha1' to option '-march='
// SSBS2: error: unsupported argument 'armv8-a+ssbs2' to option '-march='
// SVE_BF16: error: unsupported argument 'armv8-a+sve-bf16' to option '-march='
// SVE_EBF16: error: unsupported argument 'armv8-a+sve-ebf16' to option '-march='
// SVE_I8MM: error: unsupported argument 'armv8-a+sve-i8mm' to option '-march='
// SVE2_PMULL128: error: unsupported argument 'armv8-a+sve2-pmull128' to option '-march='
// DPB: error: unsupported argument 'armv8-a+dpb' to option '-march='
// RCPC2: error: unsupported argument 'armv8-a+rcpc2' to option '-march='
// FLAGM2: error: unsupported argument 'armv8-a+flagm2' to option '-march='
// FRINTTS: error: unsupported argument 'armv8-a+frintts' to option '-march='
// DPB2: error: unsupported argument 'armv8-a+dpb2' to option '-march='
