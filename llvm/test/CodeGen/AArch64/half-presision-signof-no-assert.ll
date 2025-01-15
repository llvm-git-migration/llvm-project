; RUN: llc -o - %s

; Simply check that the following does not crash (see https://github.com/llvm/llvm-project/issues/123029)

target triple = "aarch64-unknown-linux-gnu"

define noalias noundef ptr @fn(ptr nocapture readonly %in, ptr nocapture readonly %out) local_unnamed_addr {
fn:
  %1 = load <4 x half>, ptr %in, align 16
  %2 = fcmp one <4 x half> %1, zeroinitializer
  %3 = uitofp <4 x i1> %2 to <4 x half>
  store <4 x half> %3, ptr %out, align 16

  %23 = getelementptr inbounds nuw i8, ptr %in, i64 8
  %24 = load half, ptr %23, align 4
  %25 = fcmp one half %24, 0xH0000
  %26 = uitofp i1 %25 to half
  %27 = call half @llvm.copysign.f16(half %26, half %24)
  %30 = getelementptr inbounds nuw i8, ptr %out, i64 8
  store half %27, ptr %30, align 8
  ret ptr null
}
