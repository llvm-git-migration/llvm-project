//===- RISCVScheduleB.td - RISC-V Scheduling Definitions B -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Define scheduler resources associated with def operands.

/// Zvbb extension
defm "" : LMULSchedWrites<"WriteVBREV">;
defm "" : LMULSchedWrites<"WriteVCLZ">;
defm "" : LMULSchedWrites<"WriteVCPOP">;
defm "" : LMULSchedWrites<"WriteVCTZ">;
defm "" : LMULSchedWrites<"WriteVWSLL">;

/// Zvbc extension
defm "" : LMULSchedWrites<"WriteVCLMUL">;
defm "" : LMULSchedWrites<"WriteVCLMULH">;

/// Zvkb extension
defm "" : LMULSchedWrites<"WriteVANDN">;
defm "" : LMULSchedWrites<"WriteVBREV8">;
defm "" : LMULSchedWrites<"WriteVREV8">;
defm "" : LMULSchedWrites<"WriteVROL">;
defm "" : LMULSchedWrites<"WriteVROR">;

/// Zvkg extension
defm "" : LMULSchedWrites<"WriteVGHSH">;
defm "" : LMULSchedWrites<"WriteVGMUL">;

/// Zvknha or Zvknhb extensions
defm "" : LMULSchedWrites<"WriteVSHA2CH">;
defm "" : LMULSchedWrites<"WriteVSHA2CL">;
defm "" : LMULSchedWrites<"WriteVSHA2MS">;

/// Zvkned extension
defm "" : LMULSchedWrites<"WriteVAESMV">;
defm "" : LMULSchedWrites<"WriteVAESKF1">;
defm "" : LMULSchedWrites<"WriteVAESKF2">;
defm "" : LMULSchedWrites<"WriteVAESZ">;

/// Zvksed extension
defm "" : LMULSchedWrites<"WriteVSM4K">;
defm "" : LMULSchedWrites<"WriteVSM4R">;

/// Zvksh extension
defm "" : LMULSchedWrites<"WriteVSM3C">;
defm "" : LMULSchedWrites<"WriteVSM3ME">;

/// Define scheduler resources associated with use operands.
/// Zvbb extension
defm "" : LMULSchedReads<"ReadVBREV">;
defm "" : LMULSchedReads<"ReadVCLZ">;
defm "" : LMULSchedReads<"ReadVCPOP">;
defm "" : LMULSchedReads<"ReadVCTZ">;
defm "" : LMULSchedReads<"ReadVWSLL">;

/// Zvbc extension
defm "" : LMULSchedReads<"ReadVCLMUL">;
defm "" : LMULSchedReads<"ReadVCLMULH">;

/// Zvkb extension
defm "" : LMULSchedReads<"ReadVANDN">;
defm "" : LMULSchedReads<"ReadVBREV8">;
defm "" : LMULSchedReads<"ReadVREV8">;
defm "" : LMULSchedReads<"ReadVROL">;
defm "" : LMULSchedReads<"ReadVROR">;

/// Zvkg extension
defm "" : LMULSchedReads<"ReadVGHSH">;
defm "" : LMULSchedReads<"ReadVGMUL">;

/// Zvknha or Zvknhb extensions
defm "" : LMULSchedReads<"ReadVSHA2CH">;
defm "" : LMULSchedReads<"ReadVSHA2CL">;
defm "" : LMULSchedReads<"ReadVSHA2MS">;

/// Zvkned extension
defm "" : LMULSchedReads<"ReadVAESMV">;
defm "" : LMULSchedReads<"ReadVAESKF1">;
defm "" : LMULSchedReads<"ReadVAESKF2">;
defm "" : LMULSchedReads<"ReadVAESZ">;

/// Zvksed extension
defm "" : LMULSchedReads<"ReadVSM4K">;
defm "" : LMULSchedReads<"ReadVSM4R">;

/// Zvksh extension
defm "" : LMULSchedReads<"ReadVSM3C">;
defm "" : LMULSchedReads<"ReadVSM3ME">;

multiclass UnsupportedSchedZvbb {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVBREV", []>;
defm "" : LMULWriteRes<"WriteVCLZ", []>;
defm "" : LMULWriteRes<"WriteVCPOP", []>;
defm "" : LMULWriteRes<"WriteVCTZ", []>;
defm "" : LMULWriteRes<"WriteVWSLL", []>;

defm "" : LMULReadAdvance<"ReadVBREV", 0>;
defm "" : LMULReadAdvance<"ReadVCLZ", 0>;
defm "" : LMULReadAdvance<"ReadVCPOP", 0>;
defm "" : LMULReadAdvance<"ReadVCTZ", 0>;
defm "" : LMULReadAdvance<"ReadVWSLL", 0>;
}
}

multiclass UnsupportedSchedZvbc {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVCLMUL", []>;
defm "" : LMULWriteRes<"WriteVCLMULH", []>;

defm "" : LMULReadAdvance<"ReadVCLMUL", 0>; 
defm "" : LMULReadAdvance<"ReadVCLMULH", 0>;
}
}

multiclass UnsupportedSchedZvkb {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVANDN", []>;
defm "" : LMULWriteRes<"WriteVBREV8", []>;
defm "" : LMULWriteRes<"WriteVREV8", []>;
defm "" : LMULWriteRes<"WriteVROL", []>;
defm "" : LMULWriteRes<"WriteVROR", []>;

defm "" : LMULReadAdvance<"ReadVANDN", 0>;
defm "" : LMULReadAdvance<"ReadVBREV8", 0>;
defm "" : LMULReadAdvance<"ReadVREV8", 0>;
defm "" : LMULReadAdvance<"ReadVROL", 0>;
defm "" : LMULReadAdvance<"ReadVROR", 0>;
}
}

multiclass UnsupportedSchedZvkg {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVGHSH", []>;
defm "" : LMULWriteRes<"WriteVGMUL", []>;

defm "" : LMULReadAdvance<"ReadVGHSH", 0>;
defm "" : LMULReadAdvance<"ReadVGMUL", 0>;
}
}

multiclass UnsupportedSchedZvknhaOrZvknhb {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVSHA2CH", []>;
defm "" : LMULWriteRes<"WriteVSHA2CL", []>;
defm "" : LMULWriteRes<"WriteVSHA2MS", []>;

defm "" : LMULReadAdvance<"ReadVSHA2CH", 0>;
defm "" : LMULReadAdvance<"ReadVSHA2CL", 0>;
defm "" : LMULReadAdvance<"ReadVSHA2MS", 0>;
}
}

multiclass UnsupportedSchedZvkned {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVAESMV", []>;
defm "" : LMULWriteRes<"WriteVAESKF1", []>;
defm "" : LMULWriteRes<"WriteVAESKF2", []>;
defm "" : LMULWriteRes<"WriteVAESZ", []>;

defm "" : LMULReadAdvance<"ReadVAESMV", 0>;
defm "" : LMULReadAdvance<"ReadVAESKF1", 0>;
defm "" : LMULReadAdvance<"ReadVAESKF2", 0>;
defm "" : LMULReadAdvance<"ReadVAESZ", 0>;
}
}

multiclass UnsupportedSchedZvksed {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVSM4K", []>;
defm "" : LMULWriteRes<"WriteVSM4R", []>;

defm "" : LMULReadAdvance<"ReadVSM4K", 0>;
defm "" : LMULReadAdvance<"ReadVSM4R", 0>;
}
}

multiclass UnsupportedSchedZvksh {
let Unsupported = true in {
defm "" : LMULWriteRes<"WriteVSM3C", []>;
defm "" : LMULWriteRes<"WriteVSM3ME", []>;

defm "" : LMULReadAdvance<"ReadVSM3C", 0>;
defm "" : LMULReadAdvance<"ReadVSM3ME", 0>;
}
}

// Helper class to define all RISC-V Vector Crypto extensions as unsupported
multiclass UnsupportedSchedZvk {
defm "" : UnsupportedSchedZvbb;
defm "" : UnsupportedSchedZvbc;
defm "" : UnsupportedSchedZvkb;
defm "" : UnsupportedSchedZvkg;
defm "" : UnsupportedSchedZvknhaOrZvknhb;
defm "" : UnsupportedSchedZvkned;
defm "" : UnsupportedSchedZvksed;
defm "" : UnsupportedSchedZvksh;
}
