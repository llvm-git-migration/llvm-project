// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:     -Wa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-MMSA
// CHECK-MMSA: "-cc1" {{.*}}"-mmsa"

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:     -Wa,-mno-msa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-MMSA

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:     -Wa,-mmsa,-mno-msa %s -Werror 2>&1 | FileCheck %s --check-prefix=CHECK-NOMMSA
// CHECK-NOMMSA: "-cc1"

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:    -fno-integrated-as -Wa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=MIPS-MSA
// MIPS-MSA: as{{(.exe)?}}" "-march" "mips64r2" "-mabi" "64" "-EL" "-KPIC" "-mmsa"

// RUN: %clang -### -c --target=mips64el-unknown-linux-gnuabi64 \
// RUN:    -fno-integrated-as -Wa,-mno-msa %s -Werror 2>&1 | FileCheck %s --check-prefix=MIPS-NOMSA
// MIPS-NOMSA: as{{(.exe)?}}"
// MIPS-NOMSA-NOT: "-mmsa"

// RUN: not %clang -### -c --target=x86_64-unknown-linux-gnu \
// RUN:     -Wa,-mmsa %s -Werror 2>&1 | FileCheck %s --check-prefix=ERR-MSA
// ERR-MSA: error: unsupported option '-Wa,-mmsa' for target '{{.*}}'

// RUN: not %clang -### -c --target=x86_64-unknown-linux-gnu \
// RUN:     -Wa,-mno-msa %s -Werror 2>&1 | FileCheck %s --check-prefix=ERR-NOMSA
// ERR-NOMSA: error: unsupported option '-Wa,-mno-msa' for target '{{.*}}'
