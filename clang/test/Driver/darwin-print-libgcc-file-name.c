// Test the output of -print-libgcc-file-name on Darwin.

//
// All platforms
//

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=x86_64-apple-macos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-MACOS %s
// CHECK-CLANGRT-MACOS: libclang_rt.osx.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-ios \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-IOS %s
// CHECK-CLANGRT-IOS: libclang_rt.ios.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-watchos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-WATCHOS %s
// CHECK-CLANGRT-WATCHOS: libclang_rt.watchos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-tvos \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-TVOS %s
// CHECK-CLANGRT-TVOS: libclang_rt.tvos.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=arm64-apple-driverkit \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-DRIVERKIT %s
// CHECK-CLANGRT-DRIVERKIT: libclang_rt.driverkit.a

// TODO add simulators

//
// Check the cc_kext variants
//

// TODO

//
// Check the sanitizer and profile variants
//

// TODO

//
// Check the dynamic library variants
//

// TODO