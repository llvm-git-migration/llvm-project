// UNSUPPORTED: system-linux

// RUN: %clang -### -nogpulib -nogpuinc \
// RUN:  --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK-NOT: ROCm installation search path: [[ROCM_PATH:.*/usr$]]
