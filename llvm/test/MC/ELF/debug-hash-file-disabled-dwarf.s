// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -o %t %s
// RUN: llvm-readelf --sections %t | FileCheck %s

// CHECK: Section Headers:
// CHECK-NOT: .debug_

# 1 "/MyTest/Inputs/other.S"

foo:
  nop
  nop
  nop
