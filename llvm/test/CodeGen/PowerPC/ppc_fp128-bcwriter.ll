; RUN: llvm-as < %s -o - | llvm-dis - | FileCheck %s

;CHECK-LABEL: main
;CHECK: store ppc_fp128 f0xFFFFFFFFFFFFFFFF0000000000000000

define i32 @main() local_unnamed_addr {
_main_entry:
  %e3 = alloca ppc_fp128, align 16
  store ppc_fp128 f0xFFFFFFFFFFFFFFFF0000000000000000, ptr %e3, align 16
  %0 = call i64 @foo( ptr nonnull %e3)
  ret i32 undef
}

declare i64 @foo(ptr)

