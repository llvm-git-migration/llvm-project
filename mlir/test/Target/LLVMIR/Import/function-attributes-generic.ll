; RUN: mlir-translate -import-llvm -split-input-file %s --mlir-print-op-generic | FileCheck %s

; Ensure that no empty parameter attribute lists are created.
; CHECK: "llvm.func"
; CHECK-SAME:  <{
; CHECK-NOT:  arg_attr
; CHECK-NOT:  res_attrs
; CHECK-SAME:  }>
declare ptr @func_no_param_attrs()

; Ensure that we have dso_local
; CHECK: "llvm.func"()
; CHECK-SAME: <{
; CHECK-SAME: dso_local
; CHECK-SAME: "dsolocal_func"
; CHECK-SAME:  }>
define dso_local void @dsolocal_func() {
  ret void
}
