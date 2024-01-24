; RUN: llc -mtriple=riscv32 -mattr=+experimental-zfbfmin -verify-machineinstrs \
; RUN:   -target-abi ilp32f < %s | FileCheck -check-prefixes=CHECK32ZFBFMIN,RV32IZFBFMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+d,+experimental-zfbfmin -verify-machineinstrs \
; RUN:   -target-abi ilp32d < %s | FileCheck -check-prefixes=CHECK32ZFBFMIN,R32IDZFBFMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+d -verify-machineinstrs \
; RUN:   -target-abi ilp32d < %s | FileCheck -check-prefixes=RV32ID %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zfbfmin -verify-machineinstrs \
; RUN:   -target-abi lp64f < %s | FileCheck -check-prefixes=CHECK64ZFBFMIN,RV64IZFBFMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+d,+experimental-zfbfmin -verify-machineinstrs \
; RUN:   -target-abi lp64d < %s | FileCheck -check-prefixes=CHECK64ZFBFMIN,RV64IDZFBFMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+d -verify-machineinstrs \
; RUN:   -target-abi lp64d < %s | FileCheck -check-prefixes=RV64ID %s

define signext i8 @test_floor_si8(bfloat %x) {
  %a = call bfloat @llvm.floor.bf16(bfloat %x)
  %b = fptosi bfloat %a to i8
  ret i8 %b
}

; define signext i16 @test_floor_si16(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_floor_si32(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_floor_si64(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i64
;   ret i64 %b
; }

; define zeroext i8 @test_floor_ui8(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i8
;   ret i8 %b
; }

; define zeroext i16 @test_floor_ui16(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_floor_ui32(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_floor_ui64(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i64
;   ret i64 %b
; }

; define signext i8 @test_ceil_si8(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i8
;   ret i8 %b
; }

; define signext i16 @test_ceil_si16(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_ceil_si32(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_ceil_si64(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i64
;   ret i64 %b
; }

; define zeroext i8 @test_ceil_ui8(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i8
;   ret i8 %b
; }

; define zeroext i16 @test_ceil_ui16(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_ceil_ui32(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_ceil_ui64(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i64
;   ret i64 %b
; }

; define signext i8 @test_trunc_si8(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i8
;   ret i8 %b
; }

; define signext i16 @test_trunc_si16(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_trunc_si32(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_trunc_si64(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i64
;   ret i64 %b
; }

; define zeroext i8 @test_trunc_ui8(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i8
;   ret i8 %b
; }

; define zeroext i16 @test_trunc_ui16(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_trunc_ui32(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_trunc_ui64(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i64
;   ret i64 %b
; }

; define signext i8 @test_round_si8(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i8
;   ret i8 %b
; }

; define signext i16 @test_round_si16(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_round_si32(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_round_si64(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i64
;   ret i64 %b
; }

; define zeroext i8 @test_round_ui8(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i8
;   ret i8 %b
; }

; define zeroext i16 @test_round_ui16(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_round_ui32(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_round_ui64(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i64
;   ret i64 %b
; }

; define signext i8 @test_roundeven_si8(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i8
;   ret i8 %b
; }

; define signext i16 @test_roundeven_si16(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_roundeven_si32(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_roundeven_si64(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptosi bfloat %a to i64
;   ret i64 %b
; }

; define zeroext i8 @test_roundeven_ui8(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i8
;   ret i8 %b
; }

; define zeroext i16 @test_roundeven_ui16(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i16
;   ret i16 %b
; }

; define signext i32 @test_roundeven_ui32(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i32
;   ret i32 %b
; }

; define i64 @test_roundeven_ui64(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   %b = fptoui bfloat %a to i64
;   ret i64 %b
; }

; define bfloat @test_floor_bfloat(bfloat %x) {
;   %a = call bfloat @llvm.floor.bf16(bfloat %x)
;   ret bfloat %a
; }

; define bfloat @test_ceil_bfloat(bfloat %x) {
;   %a = call bfloat @llvm.ceil.bf16(bfloat %x)
;   ret bfloat %a
; }

; define bfloat @test_trunc_bfloat(bfloat %x) {
;   %a = call bfloat @llvm.trunc.bf16(bfloat %x)
;   ret bfloat %a
; }

; define bfloat @test_round_bfloat(bfloat %x) {
;   %a = call bfloat @llvm.round.bf16(bfloat %x)
;   ret bfloat %a
; }

; define bfloat @test_roundeven_bfloat(bfloat %x) {
;   %a = call bfloat @llvm.roundeven.bf16(bfloat %x)
;   ret bfloat %a
; }

declare bfloat @llvm.floor.bf16(bfloat)
; declare bfloat @llvm.ceil.bf16(bfloat)
; declare bfloat @llvm.trunc.bf16(bfloat)
; declare bfloat @llvm.round.bf16(bfloat)
; declare bfloat @llvm.roundeven.bf16(bfloat)
