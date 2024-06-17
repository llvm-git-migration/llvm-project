; RUN: opt -S -passes=sroa %s | FileCheck %s
;
;; Check that we do not assert and that we retain the fake_use instruction that
;; uses the address of bar.
;
; CHECK: define{{.*}}foo
; CHECK: call{{.*llvm\.fake\.use.*}}%bar.addr

define void @_Z3fooPi(i32* %bar) {
entry:
  %bar.addr = alloca i32*, align 8
  %baz = alloca i8**, align 8
  store i32* %bar, i32** %bar.addr, align 8
  %0 = bitcast i32** %bar.addr to i8**
  store i8** %0, i8*** %baz, align 8
  %1 = load i32*, i32** %bar.addr, align 8
  call void (...) @llvm.fake.use(i32* %1)
  %2 = load i8**, i8*** %baz, align 8
  call void (...) @llvm.fake.use(i8** %2)
  ret void
}

declare void @llvm.fake.use(...)
