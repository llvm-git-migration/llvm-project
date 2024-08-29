; REQUIRES: asserts

; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -debug -amdgpu-module-splitting-no-externalize-address-taken 2>&1 | FileCheck %s

; CHECK:      [!] callgraph is incomplete for A - analyzing function
; CHECK-NEXT:    resolved call to PerryThePlatypus in:   call void @Perry()

@Perry = internal alias ptr(), ptr @PerryThePlatypus

define internal void @PerryThePlatypus() {
  ret void
}

define amdgpu_kernel void @A() {
  call void @Perry()
  ret void
}

define amdgpu_kernel void @B() {
  call void @PerryThePlatypus()
  ret void
}
