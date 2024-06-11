/// This test uses the PATH environment variable; on Windows, we may need to retain
/// the original path for the built Clang binary to be able to execute (as it is
/// used for locating dependent DLLs).
// UNSUPPORTED: system-windows

//-----------------------------------------------------------------------------
// Check llvm-spirv-<LLVM_VERSION_MAJOR> is used if it is found in PATH.
// RUN: mkdir -p %t/versioned
// RUN: touch %t/versioned/llvm-spirv-%llvm-version-major \
// RUN:   && chmod +x %t/versioned/llvm-spirv-%llvm-version-major
// RUN: env "PATH=%t/versioned" %clang -### --target=spirv64 -x cl -c %s 2>&1 \
// RUN:   | FileCheck -DVERSION=%llvm-version-major --check-prefix=VERSIONED %s

// VERSIONED: {{.*}}llvm-spirv-[[VERSION]]
