// RUN: %clang -### --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 -MD -MF tmp.d %s 2>&1 | FileCheck %s

// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1030"{{.*}}"-dependency-file" "tmp.d"
// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1100"{{.*}}"-dependency-file" "tmp.d"
// CHECK-NOT: {{.*}}clang{{.*}}"-target-cpu" "gfx1101"{{.*}}"-dependency-file" "tmp.d"
// CHECK: {{.*}}clang{{.*}}"-target-cpu" "x86-64"{{.*}}"-dependency-file" "tmp.d"
