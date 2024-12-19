// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `/` in the test.

// Add default directories before running clang to check default
// search paths.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp -R %S/Inputs/MacOSX15.1.sdk %t/
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/include
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/include/c++/v1
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/local/include
// RUN: mkdir -p %t/MacOSX15.1.sdk/embedded/usr/include
// RUN: mkdir -p %t/MacOSX15.1.sdk/embedded/usr/local/include

// Unlike the Darwin driver, the MachO driver doesn't add any framework search paths,
// only the normal header ones.
// RUN: %clang -xc %s -target arm64-apple-none-macho -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck --check-prefix=CHECK-C %s
//
// CHECK-C:                 -isysroot [[SDKROOT:[^ ]*/MacOSX15.1.sdk]]
// CHECK-C:                 #include <...> search starts here:
// CHECK-C-NEXT:            [[SDKROOT]]/usr/local/include
// CHECK-C-NEXT:            /clang/{{.*}}/include
// CHECK-C-NEXT:            [[SDKROOT]]/usr/include

// Unlike the Darwin driver, the MachO driver doesn't default to libc++
// RUN: %clang -xc++ %s -target arm64-apple-none-macho -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck --check-prefix=CHECK-CXX %s
//
// CHECK-CXX:               -isysroot [[SDKROOT:[^ ]*/MacOSX15.1.sdk]]
// CHECK-CXX:               #include <...> search starts here:
// CHECK-CXX-NEXT:          [[SDKROOT]]/usr/local/include
// CHECK-CXX-NEXT:          /clang/{{.*}}/include
// CHECK-CXX-NEXT:          [[SDKROOT]]/usr/include

// However, if the user requests libc++, the MachO driver should find the search path.
// RUN: %clang -xc++ -stdlib=libc++ %s -target arm64-apple-none-macho -isysroot %t/MacOSX15.1.sdk -E -v 2>&1 | FileCheck --check-prefix=CHECK-LIBCXX %s
//
// CHECK-LIBCXX:            -isysroot [[SDKROOT:[^ ]*/MacOSX15.1.sdk]]
// CHECK-LIBCXX:            #include <...> search starts here:
// CHECK-LIBCXX-NEXT:       [[SDKROOT]]/usr/include/c++/v1
// CHECK-LIBCXX-NEXT:       [[SDKROOT]]/usr/local/include
// CHECK-LIBCXX-NEXT:       /clang/{{.*}}/include
// CHECK-LIBCXX-NEXT:       [[SDKROOT]]/usr/include

// Verify that embedded uses can swap in alternate usr/include and usr/local/include directories.
// usr/local/include is specified in the driver as -internal-isystem, however, the driver generated
// paths come before the paths in the driver arguments. In order to keep usr/local/include in the
// same position, -isystem has to be used instead of -Xclang -internal-isystem. There isn't an
// -externc-isystem, but it's ok to use -Xclang -internal-externc-isystem since the driver doesn't
// use that if -nostdlibinc or -nostdinc is passed.
// RUN: %clang -xc++ -stdlib=libc++ %s -target arm64-apple-none-macho -isysroot %t/MacOSX15.1.sdk -nostdlibinc -isystem %t/MacOSX15.1.sdk/embedded/usr/local/include -Xclang -internal-externc-isystem -Xclang %t/MacOSX15.1.sdk/embedded/usr/include -E -v 2>&1 | FileCheck --check-prefix=CHECK-EMBEDDED %s
//
// CHECK-EMBEDDED:          -isysroot [[SDKROOT:[^ ]*/MacOSX15.1.sdk]]
// CHECK-EMBEDDED:          #include <...> search starts here:
// CHECK-EMBEDDED-NEXT:     [[SDKROOT]]/embedded/usr/local/include
// CHECK-EMBEDDED-NEXT:     /clang/{{.*}}/include
// CHECK-EMBEDDED-NEXT:     [[SDKROOT]]/embedded/usr/include
