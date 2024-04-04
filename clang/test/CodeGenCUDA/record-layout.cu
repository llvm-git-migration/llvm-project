// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdump-record-layouts \
// RUN:   -emit-llvm -o %t -xhip %s 2>&1 | FileCheck %s --check-prefix=AST
// RUN: cat %t | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx1100 \
// RUN:   -emit-llvm -fdump-record-layouts -aux-triple x86_64-pc-windows-msvc \
// RUN:   -o %t -xhip %s | FileCheck %s --check-prefix=AST
// RUN: cat %t | FileCheck %s

#include "Inputs/cuda.h"

// AST: *** Dumping AST Record Layout
// AST-LABEL:         0 | struct C
// AST-NEXT:          0 |   struct A (base) (empty)
// AST-NEXT:          1 |   struct B (base) (empty)
// AST-NEXT:          4 |   int i
// AST-NEXT:            | [sizeof=8, align=4,
// AST-NEXT:            |  nvsize=8, nvalign=4]

// CHECK: %struct.C = type { [4 x i8], i32 }

struct A {};
struct B {};
struct C : A, B {
    int i;
};

__device__ C c;
__global__ void test_C(C c)
{}
 
// AST: *** Dumping AST Record Layout
// AST-LABEL:          0 | struct I
// AST-NEXT:           0 |   (I vftable pointer)
// AST-NEXT:           8 |   int i
// AST-NEXT:             | [sizeof=16, align=8,
// AST-NEXT:             |  nvsize=16, nvalign=8]

// AST: *** Dumping AST Record Layout
// AST-LABEL:          0 | struct J
// AST-NEXT:           0 |   struct I (primary base)
// AST-NEXT:           0 |     (I vftable pointer)
// AST-NEXT:           8 |     int i
// AST-NEXT:          16 |   int j
// AST-NEXT:             | [sizeof=24, align=8,
// AST-NEXT:             |  nvsize=24, nvalign=8]

// CHECK: %struct.J = type { %struct.I, i32 }
// CHECK: %struct.I = type { ptr, i32 }

struct I {
    virtual void f() = 0;
    int i;
};
struct J : I {
    void f() override {}
    int j;
};

__global__ void test_J(J j)
{}

void test(C c, J j) {
  test_C<<<1, 1>>>(c);
  test_J<<<1, 1>>>(j); 
}
