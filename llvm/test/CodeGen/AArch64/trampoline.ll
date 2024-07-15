; RUN: llc -mtriple=aarch64-- < %s | FileCheck %s
; UNSUPPORTED: darwin, system-windows

declare void @llvm.init.trampoline(i8*, i8*, i8*);
declare i8* @llvm.adjust.trampoline(i8*);

define i64 @func(i64* nest %ptr, i64 %val)
{
    %x = load i64, i64* %ptr
    %sum = add i64 %x, %val
    ret i64 %sum
}

; CHECK-LABEL: main
define i64 @main(i64, i8**)
{
    %val = alloca i64
    store i64 13, i64* %val
    %nval = bitcast i64* %val to i8*
    %tramp_buf = alloca [36 x i8], align 4
    %tramp = getelementptr [36 x i8], [36 x i8]* %tramp_buf, i64 0, i64 0
; CHECK:	bl	__trampoline_setup
    call void @llvm.init.trampoline(
            i8* %tramp,
            i8* bitcast (i64 (i64*, i64)* @func to i8*),
            i8* %nval)
    %ptr = call i8* @llvm.adjust.trampoline(i8* %tramp)
    %fptr = bitcast i8* %ptr to i64(i64)*
    %retval = call i64 %fptr (i64 42)
    ret i64 %retval
}
