// RUN: clang -x hip -O3 -fgpu-rdc %s -mllvm -debug-only=amdgpu-attributor -o - | FileCheck %s

// CHECK: Module {{.*}} is not assumed to be a closed world
// CHECK: Module ld-temp.o is assumed to be a closed world

__attribute__((device)) int foo() {
    return 1;
}

int main() {
    return 0;
}
