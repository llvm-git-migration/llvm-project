// Default cf-branch-label-scheme is func-sig
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -fcf-protection=branch -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-FUNC-SIG %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -fcf-protection=branch -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-FUNC-SIG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=unlabeled -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-UNLABELED %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=func-sig -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-FUNC-SIG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-FUNC-SIG-SCHEME-UNUSED %s

// RUN: not %clang --target=riscv32 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=unlabeled -E -dM %s -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-BRANCH-PROT-INVALID %s

// RUN: not %clang --target=riscv32 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=func-sig -E -dM %s -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-BRANCH-PROT-INVALID %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-FUNC-SIG-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=unlabeled -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-UNLABELED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=func-sig -E -dM %s -o - \
// RUN:   | FileCheck --check-prefix=CHECK-ZICFILP-FUNC-SIG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-FUNC-SIG-SCHEME-UNUSED %s

// RUN: not %clang --target=riscv64 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=unlabeled -E -dM %s -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-BRANCH-PROT-INVALID %s

// RUN: not %clang --target=riscv64 -fcf-protection=branch \
// RUN:   -mcf-branch-label-scheme=func-sig -E -dM %s -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-BRANCH-PROT-INVALID %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN:   -o - 2>&1 | FileCheck \
// RUN:   --check-prefixes=CHECK-NO-MACRO,CHECK-FUNC-SIG-SCHEME-UNUSED %s

// CHECK-ZICFILP-UNLABELED: __riscv_landing_pad 1{{$}}
// CHECK-ZICFILP-UNLABELED: __riscv_landing_pad_unlabeled 1{{$}}
// CHECK-ZICFILP-FUNC-SIG: __riscv_landing_pad 1{{$}}
// CHECK-ZICFILP-FUNC-SIG: __riscv_landing_pad_func_sig 1{{$}}
// CHECK-NO-MACRO-NOT: __riscv_landing_pad
// CHECK-NO-MACRO-NOT: __riscv_landing_pad_unlabeled
// CHECK-NO-MACRO-NOT: __riscv_landing_pad_func_sig
// CHECK-BRANCH-PROT-INVALID: error: option 'cf-protection=branch' cannot be
// CHECK-BRANCH-PROT-INVALID-SAME: specified on this target
// CHECK-UNLABELED-SCHEME-UNUSED: warning: argument unused during compilation:
// CHECK-UNLABELED-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=unlabeled'
// CHECK-FUNC-SIG-SCHEME-UNUSED: warning: argument unused during compilation:
// CHECK-FUNC-SIG-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=func-sig'
