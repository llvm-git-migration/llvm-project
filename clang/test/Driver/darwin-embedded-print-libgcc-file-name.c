// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7em-apple-darwin \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-HARD_STATIC %s
// CHECK-CLANGRT-HARD_STATIC: libclang_rt.hard_static.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7em-apple-darwin -msoft-float \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-SOFT_STATIC %s
// CHECK-CLANGRT-SOFT_STATIC: libclang_rt.soft_static.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7em-apple-darwin -fPIC \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-HARD_PIC %s
// CHECK-CLANGRT-HARD_PIC: libclang_rt.hard_pic.a

// RUN: %clang -rtlib=compiler-rt -print-libgcc-file-name \
// RUN:     --target=armv7em-apple-darwin -msoft-float -fPIC \
// RUN:     -resource-dir=%S/Inputs/resource_dir 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGRT-SOFT_PIC %s
// CHECK-CLANGRT-SOFT_PIC: libclang_rt.soft_pic.a

// FIXME: -print-libgcc-file-name is using the default toolchain
//        so the tests above do not give the right answer yet.
// XFAIL: *
