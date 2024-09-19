// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s  -DFILE=%s

// CHECK: [[FILE]]:[[@LINE+2]]:33: error: expected list of list type argument in !listflatten operator
class Flatten<list<int> A> {
    list<int> F = !listflatten(A);
}
