// Use --implicit-check-not={{[a-zA-Z0-9]}} to ensure no additional CPUs are in this list

// RUN: not %clang_cc1 -triple hexagon--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not={{[a-zA-Z0-9]}}
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: hexagonv5,
// CHECK-SAME: hexagonv55,
// CHECK-SAME: hexagonv60,
// CHECK-SAME: hexagonv62,
// CHECK-SAME: hexagonv65,
// CHECK-SAME: hexagonv66,
// CHECK-SAME: hexagonv67,
// CHECK-SAME: hexagonv67t,
// CHECK-SAME: hexagonv68,
// CHECK-SAME: hexagonv69,
// CHECK-SAME: hexagonv71,
// CHECK-SAME: hexagonv71t,
// CHECK-SAME: hexagonv73
// CHECK-SAME: {{$}}
