; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-no-externalize-address-taken
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s

; 3 kernels:
;   - A calls nothing
;   - B calls @PerryThePlatypus
;   - C calls @Perry, an alias of @PerryThePlatypus
;
; We should see through the alias and treat C as-if it called PerryThePlatypus directly.

; CHECK0: define internal void @PerryThePlatypus()
; CHECK0: define amdgpu_kernel void @C

; CHECK1: define internal void @PerryThePlatypus()
; CHECK1: define amdgpu_kernel void @A
; CHECK1: define amdgpu_kernel void @B

@Perry = internal alias ptr(), ptr @PerryThePlatypus

define internal void @PerryThePlatypus() {
  ret void
}

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B() {
  call void @PerryThePlatypus()
  ret void
}

define amdgpu_kernel void @C() {
  call void @Perry()
  ret void
}
