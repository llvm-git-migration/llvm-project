// RUN: not %clang --target=riscv32 -fcf-protection=branch -c %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BRANCH-PROT-INVALID %s

// RUN: not %clang --target=riscv32 -fcf-protection=branch -c %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BRANCH-PROT-INVALID %s

// RUN: not %clang --target=riscv64 -fcf-protection=branch -c %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BRANCH-PROT-INVALID %s

// RUN: not %clang --target=riscv64 -fcf-protection=branch -c %s \
// RUN:   -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BRANCH-PROT-INVALID %s

// CHECK-BRANCH-PROT-INVALID: error: option 'cf-protection=branch' cannot be
// CHECK-BRANCH-PROT-INVALID-SAME: specified on this target
