
_Static_assert(0, "");

/// Test that piping the output into another process disables syntax
/// highlighting of code snippets.

// RUN: not %clang_cc1 %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: static assertion failed:
// CHECK-NEXT: {{^}}   2 | _Static_assert(0, "");{{$}}



// RUN: not %clang_cc1 %s -o /dev/null -fcolor-diagnostics > %t 2>&1
// RUN: FileCheck -check-prefix=COLOR --input-file=%t %s
// COLOR: error: static assertion failed:
// COLOR-NEXT: ^[[0;
