; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=ASSEM-DISASS
; RUN: opt < %s -O3 -S | FileCheck %s --check-prefix=OPT
; RUN: verify-uselistorder %s
; Basic smoke tests for bfloat type.

define bfloat @check_bfloat(bfloat %A) {
; ASSEM-DISASS: ret bfloat %A
    ret bfloat %A
}

define bfloat @check_bfloat_literal() {
; ASSEM-DISASS: ret bfloat f0x3149
    ret bfloat f0x3149
}

define <4 x bfloat> @check_fixed_vector() {
; ASSEM-DISASS: ret <4 x bfloat> %tmp
  %tmp = fadd <4 x bfloat> undef, undef
  ret <4 x bfloat> %tmp
}

define <vscale x 4 x bfloat> @check_vector() {
; ASSEM-DISASS: ret <vscale x 4 x bfloat> %tmp
  %tmp = fadd <vscale x 4 x bfloat> undef, undef
  ret <vscale x 4 x bfloat> %tmp
}

define bfloat @check_bfloat_constprop() {
  %tmp = fadd bfloat f0x40C0, f0x40C0
; OPT: f0x4140
  ret bfloat %tmp
}

define float @check_bfloat_convert() {
  %tmp = fpext bfloat f0x4C8D to float
; OPT: 0x4191A00000000000
  ret float %tmp
}

; ASSEM-DISASS-LABEL @snan_bfloat
define bfloat @snan_bfloat() {
; ASSEM-DISASS: ret bfloat f0x7F81
    ret bfloat f0x7F81
}

; ASSEM-DISASS-LABEL @qnan_bfloat
define bfloat @qnan_bfloat() {
; ASSEM-DISASS: ret bfloat f0x7FC0
    ret bfloat f0x7FC0
}

; ASSEM-DISASS-LABEL @pos_inf_bfloat
define bfloat @pos_inf_bfloat() {
; ASSEM-DISASS: ret bfloat f0x7F80
    ret bfloat f0x7F80
}

; ASSEM-DISASS-LABEL @neg_inf_bfloat
define bfloat @neg_inf_bfloat() {
; ASSEM-DISASS: ret bfloat f0xFF80
    ret bfloat f0xFF80
}
