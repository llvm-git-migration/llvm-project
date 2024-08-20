// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:   -Wa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-MMSA
// CHECK-MMSA: "-cc1" {{.*}}"-mmsa"

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:   -Wa,-mmsa,-mno-msa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-NOMMSA
// CHECK-NOMMSA:     "-cc1"
// CHECK-NOMMSA-NOT: "-mssa"
