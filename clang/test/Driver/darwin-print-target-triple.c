// Test the output of -print-target-triple on Darwin.

//
// All platforms
//

// RUN: %clang -print-target-triple \
// RUN:     --target=x86_64-apple-macos -mmacos-version-min=15 \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-MACOS %s
// CHECK-CLANGRT-MACOS: x86_64-apple-macosx15.0.0

// RUN: %clang -print-target-triple \
// RUN:     --target=arm64-apple-ios -mios-version-min=9 \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-IOS %s
// CHECK-CLANGRT-IOS: arm64-apple-ios9.0.0

// RUN: %clang -print-target-triple \
// RUN:     --target=arm64-apple-watchos -mwatchos-version-min=3 \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-WATCHOS %s
// CHECK-CLANGRT-WATCHOS: arm64-apple-watchos3.0.0

// RUN: %clang -print-target-triple \
// RUN:     --target=arm64-apple-tvos -mtvos-version-min=1\
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-TVOS %s
// CHECK-CLANGRT-TVOS: arm64-apple-tvos1.0.0

// RUN: %clang -print-target-triple \
// RUN:     --target=arm64-apple-driverkit \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-DRIVERKIT %s
// CHECK-CLANGRT-DRIVERKIT: arm64-apple-driverkit19.0.0
