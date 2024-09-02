; RUN: llc -verify-machineinstrs < %s -relocation-model=pic

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Passing a pointer to thread-local storage to a function can be problematic
; since computing such addresses requires a function call that is introduced
; very late in instruction selection. We need to ensure that we don't introduce
; nested call sequence markers if this function call happens in a call sequence.

@TLS = internal thread_local global i64 zeroinitializer, align 8
declare void @bar(ptr)
define internal void @foo() {
call void @bar(ptr @TLS)
call void @bar(ptr @TLS)
ret void
}