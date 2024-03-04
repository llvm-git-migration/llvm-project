; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -start-after=codegenprepare -stop-before finalize-isel | FileCheck %s

define <8 x i16> @_Z3fooiPiPs(<8 x i16> %src, <8 x i1> %0) #0 {
entry:
  %res = call <8 x i16> @llvm.masked.expandload.v8i16(ptr null, <8 x i1> %0, <8 x i16> %src)
  ret <8 x i16> %res
}

; CHECK-LABEL:   name: _Z3fooiPiPs
; CHECK:         %1:vr128x = COPY $xmm1
; CHECK-NEXT:    %0:vr128x = COPY $xmm0
; CHECK-NEXT:    %2:vr128x = VPSLLWZ128ri %1, 15
; CHECK-NEXT:    %3:vk16wm = VPMOVW2MZ128rr killed %2
; CHECK-NEXT:    %4:vr128x = VPEXPANDWZ128rmk %0, killed %3, $noreg, 1, $noreg, 0, $noreg :: (load unknown-size from `ptr null`, align 16)

define <8 x i16> @_Z3foo2iPiPs(<8 x i16> %src, <8 x i1> %0) #0 {
entry:
  %res = call <8 x i16> @llvm.masked.expandload.v8i16(ptr align 32 null, <8 x i1> %0, <8 x i16> %src)
  ret <8 x i16> %res
}

; CHECK-LABEL:   name: _Z3foo2iPiPs
; CHECK:         %1:vr128x = COPY $xmm1
; CHECK-NEXT:    %0:vr128x = COPY $xmm0
; CHECK-NEXT:    %2:vr128x = VPSLLWZ128ri %1, 15
; CHECK-NEXT:    %3:vk16wm = VPMOVW2MZ128rr killed %2
; CHECK-NEXT:    %4:vr128x = VPEXPANDWZ128rmk %0, killed %3, $noreg, 1, $noreg, 0, $noreg :: (load unknown-size from `ptr null`, align 32)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare <8 x i16> @llvm.masked.expandload.v8i16(ptr, <8 x i1>, <8 x i16>)

attributes #0 = { "target-cpu"="icelake-server" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
