// Use --implicit-check-not={{[a-zA-Z0-9]}} to ensure no additional CPUs are in this list

// RUN: not %clang_cc1 -triple lanai--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not={{[a-zA-Z0-9]}}
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: v11
// CHECK-SAME: {{$}}
