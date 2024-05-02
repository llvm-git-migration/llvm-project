; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -t %t.o | FileCheck --check-prefix=SYM %s

define dso_local i32 @func() hybrid_patchable nounwind {
; SYM: [ 8](sec  4)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #func$hp_target
; CHECK-LABEL:     .def    "#func$hp_target";
; CHECK:           .section        .text,"xr",discard,"#func$hp_target"
; CHECK-NEXT:      .globl  "#func$hp_target"               // -- Begin function #func$hp_target
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#func$hp_target":                      // @"#func$hp_target"
; CHECK-NEXT:      // %bb.0:
; CHECK-NEXT:      mov     w0, #1                          // =0x1
; CHECK-NEXT:      ret
  ret i32 1
}

define void @has_varargs(...) hybrid_patchable nounwind {
; SYM: [11](sec  5)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_varargs$hp_target
; CHECK-LABEL:     .def "#has_varargs$hp_target";
; CHECK:           .section .text,"xr",discard,"#has_varargs$hp_target"
; CHECK-NEXT:      .globl  "#has_varargs$hp_target"        // -- Begin function #has_varargs$hp_target
; CHECK-NEXT:      .p2align 2
; CHECK-NEXT:  "#has_varargs$hp_target":               // @"#has_varargs$hp_target"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      sub     sp, sp, #32
; CHECK-NEXT:      stp     x0, x1, [x4, #-32]
; CHECK-NEXT:      stp     x2, x3, [x4, #-16]
; CHECK-NEXT:      add     sp, sp, #32
; CHECK-NEXT:      ret
  ret void
}

define void @has_sret(ptr sret([100 x i8])) hybrid_patchable nounwind {
; SYM: [14](sec  6)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_sret$hp_target
; CHECK-LABEL:     .def    "#has_sret$hp_target";
; CHECK:           .section        .text,"xr",discard,"#has_sret$hp_target"
; CHECK-NEXT:      .globl  "#has_sret$hp_target"           // -- Begin function #has_sret$hp_target
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#has_sret$hp_target":                  // @"#has_sret$hp_target"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      ret
  ret void
}

; hybrid_patchable attribute is ignored on internal functions
define internal i32 @static_func() hybrid_patchable nounwind {
; CHECK-LABEL:     .def    static_func;
; CHECK:       static_func:                            // @static_func
; CHECK-NEXT:      // %bb.0:
; CHECK-NEXT:      mov     w0, #2                          // =0x2
; CHECK-NEXT:      ret
  ret i32 2
}

define dso_local void @caller() nounwind {
; CHECK-LABEL:     .def    "#caller";
; CHECK:           .section        .text,"xr",discard,"#caller"
; CHECK-NEXT:      .globl  "#caller"                       // -- Begin function #caller
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#caller":                              // @"#caller"
; CHECK-NEXT:      .weak_anti_dep  caller
; CHECK-NEXT:  .set caller, "#caller"{{$}}
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:      bl      "#func"
; CHECK-NEXT:      bl      static_func
; CHECK-NEXT:      ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:      ret
  %1 = call i32 @func()
  %2 = call i32 @static_func()
  ret void
}

; CHECK-LABEL:       def    "#func$hybpatch_thunk";
; CHECK:            .section        .wowthk$aa,"xr",discard,"#func$hybpatch_thunk"
; CHECK-NEXT:       .globl  "#func$hybpatch_thunk"          // -- Begin function #func$hybpatch_thunk
; CHECK-NEXT:       .p2align        2
; CHECK-NEXT:   "#func$hybpatch_thunk":                 // @"#func$hybpatch_thunk"
; CHECK-NEXT:   .seh_proc "#func$hybpatch_thunk"
; CHECK-NEXT:   // %bb.0:
; CHECK-NEXT:       str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endprologue
; CHECK-NEXT:       adrp    x8, __os_arm64x_dispatch_call
; CHECK-NEXT:       adrp    x11, func
; CHECK-NEXT:       add     x11, x11, :lo12:func
; CHECK-NEXT:       ldr     x8, [x8, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:       adrp    x10, ($iexit_thunk$cdecl$i8$v)
; CHECK-NEXT:       add     x10, x10, :lo12:($iexit_thunk$cdecl$i8$v)
; CHECK-NEXT:       adrp    x9, "#func$hp_target"
; CHECK-NEXT:       add     x9, x9, :lo12:"#func$hp_target"
; CHECK-NEXT:       blr     x8
; CHECK-NEXT:       .seh_startepilogue
; CHECK-NEXT:       ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endepilogue
; CHECK-NEXT:       br      x11
; CHECK-NEXT:       .seh_endfunclet
; CHECK-NEXT:       .seh_endproc

; CHECK-LABEL:      .def    "#has_varargs$hybpatch_thunk";
; CHECK:            .section        .wowthk$aa,"xr",discard,"#has_varargs$hybpatch_thunk"
; CHECK-NEXT:       .globl  "#has_varargs$hybpatch_thunk"   // -- Begin function #has_varargs$hybpatch_thunk
; CHECK-NEXT:       .p2align        2
; CHECK-NEXT:"#has_varargs$hybpatch_thunk":          // @"#has_varargs$hybpatch_thunk"
; CHECK-NEXT:.seh_proc "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:// %bb.0:
; CHECK-NEXT:       str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endprologue
; CHECK-NEXT:       adrp    x8, __os_arm64x_dispatch_call
; CHECK-NEXT:       adrp    x11, has_varargs
; CHECK-NEXT:       add     x11, x11, :lo12:has_varargs
; CHECK-NEXT:       ldr     x8, [x8, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:       adrp    x10, ($iexit_thunk$cdecl$v$varargs)
; CHECK-NEXT:       add     x10, x10, :lo12:($iexit_thunk$cdecl$v$varargs)
; CHECK-NEXT:       adrp    x9, "#has_varargs$hp_target"
; CHECK-NEXT:       add     x9, x9, :lo12:"#has_varargs$hp_target"
; CHECK-NEXT:       blr     x8
; CHECK-NEXT:       .seh_startepilogue
; CHECK-NEXT:       ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:       .seh_save_reg_x x30, 16
; CHECK-NEXT:       .seh_endepilogue
; CHECK-NEXT:       br      x11
; CHECK-NEXT:       .seh_endfunclet
; CHECK-NEXT:       .seh_endproc

; CHECK-LABEL:     .def    "#has_sret$hybpatch_thunk";
; CHECK:           .section        .wowthk$aa,"xr",discard,"#has_sret$hybpatch_thunk"
; CHECK-NEXT:      .globl  "#has_sret$hybpatch_thunk"      // -- Begin function #has_sret$hybpatch_thunk
; CHECK-NEXT:      .p2align        2
; CHECK-NEXT:  "#has_sret$hybpatch_thunk":             // @"#has_sret$hybpatch_thunk"
; CHECK-NEXT:  .seh_proc "#has_sret$hybpatch_thunk"
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:      str     x30, [sp, #-16]!                // 8-byte Folded Spill
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endprologue
; CHECK-NEXT:      adrp    x9, __os_arm64x_dispatch_call
; CHECK-NEXT:      adrp    x11, has_sret
; CHECK-NEXT:      add     x11, x11, :lo12:has_sret
; CHECK-NEXT:      ldr     x12, [x9, :lo12:__os_arm64x_dispatch_call]
; CHECK-NEXT:      adrp    x10, ($iexit_thunk$cdecl$m100$v)
; CHECK-NEXT:      add     x10, x10, :lo12:($iexit_thunk$cdecl$m100$v)
; CHECK-NEXT:      adrp    x9, "#has_sret$hp_target"
; CHECK-NEXT:      add     x9, x9, :lo12:"#has_sret$hp_target"
; CHECK-NEXT:      blr     x12
; CHECK-NEXT:      .seh_startepilogue
; CHECK-NEXT:      ldr     x30, [sp], #16                  // 8-byte Folded Reload
; CHECK-NEXT:      .seh_save_reg_x x30, 16
; CHECK-NEXT:      .seh_endepilogue
; CHECK-NEXT:      br      x11
; CHECK-NEXT:      .seh_endfunclet
; CHECK-NEXT:      .seh_endproc

; Verify the hybrid bitmap
; CHECK-LABEL:     .section        .hybmp$x,"yi"
; CHECK-NEXT:      .symidx "#func$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$i8$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#has_varargs$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$v$varargs
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#has_sret$hp_target"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$m100$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx "#caller"
; CHECK-NEXT:      .symidx $ientry_thunk$cdecl$v$v
; CHECK-NEXT:      .word   1
; CHECK-NEXT:      .symidx func
; CHECK-NEXT:      .symidx $iexit_thunk$cdecl$i8$v
; CHECK-NEXT:      .word   4
; CHECK-NEXT:      .symidx "#func$hybpatch_thunk"
; CHECK-NEXT:      .symidx func
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .symidx "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:      .symidx has_varargs
; CHECK-NEXT:      .word   0
; CHECK-NEXT:      .symidx "#has_sret$hybpatch_thunk"
; CHECK-NEXT:      .symidx has_sret
; CHECK-NEXT:      .word   0

; CHECK-NEXT:      .def    func;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak  func
; CHECK-NEXT:  .set func, "EXP+#func"{{$}}
; CHECK-NEXT:      .weak  "#func"
; CHECK-NEXT:      .def    "#func";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  .set "#func", "#func$hybpatch_thunk"{{$}}
; CHECK-NEXT:      .def    has_varargs;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak   has_varargs
; CHECK-NEXT:  .set has_varargs, "EXP+#has_varargs"
; CHECK-NEXT:      .weak   "#has_varargs"
; CHECK-NEXT:      .def    "#has_varargs";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  .set "#has_varargs", "#has_varargs$hybpatch_thunk"
; CHECK-NEXT:      .def    has_sret;
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:      .weak   has_sret
; CHECK-NEXT:  .set has_sret, "EXP+#has_sret"
; CHECK-NEXT:      .weak   "#has_sret"
; CHECK-NEXT:      .def    "#has_sret";
; CHECK-NEXT:      .scl    2;
; CHECK-NEXT:      .type   32;
; CHECK-NEXT:      .endef
; CHECK-NEXT:  .set "#has_sret", "#has_sret$hybpatch_thunk"

; SYM:      [45](sec 13)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #func$hybpatch_thunk
; SYM:      [50](sec 14)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_varargs$hybpatch_thunk
; SYM:      [60](sec 16)(fl 0x00)(ty  20)(scl   2) (nx 0) 0x00000000 #has_sret$hybpatch_thunk
; SYM:      [94](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #func
; SYM-NEXT: AUX indx 45 srch 3
; SYM:      [101](sec  0)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 EXP+#func
; SYM-NEXT: [102](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 has_varargs
; SYM-NEXT: AUX indx 104 srch 3
; SYM-NEXT: [104](sec  0)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 EXP+#has_varargs
; SYM-NEXT: [105](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 has_sret
; SYM-NEXT: AUX indx 107 srch 3
; SYM-NEXT: [107](sec  0)(fl 0x00)(ty   0)(scl   2) (nx 0) 0x00000000 EXP+#has_sret
; SYM-NEXT: [108](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #has_varargs
; SYM-NEXT: AUX indx 50 srch 3
; SYM-NEXT: [110](sec  0)(fl 0x00)(ty   0)(scl  69) (nx 1) 0x00000000 #has_sret
; SYM-NEXT: AUX indx 60 srch 3
