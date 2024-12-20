//===-- RISCVSchedGenericOOO.td - Generic O3 Processor -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// We assume that:
// * 8-issue out-of-order CPU with 128 ROB entries.
// * Units:
//   * IXU (Integer GenericOOOALU Unit): 4 units, only one can execute division.
//   * FXU (Floating-point Unit): 2 units.
//   * LSU (Load/Store Unit): 2 units.
//   * VXU (Vector Unit): 1 unit.
// * Latency:
//   * Integer instructions: 1 cycle.
//   * Multiplication instructions: 4 cycles.
//   * Multiplication/Division instructions: 7-13 cycles.
//   * Floating-point instructions: 4-6 cycles.
//   * Vector instructions: 2-6 cycles.
//   * Load/Store:
//     * IXU: 4 cycles.
//     * FXU: 6 cycles.
//     * VXU: 6 cycles.
// * Integer/floating-point/vector div/rem/sqrt/... are non-pipelined.
// * 
//===----------------------------------------------------------------------===//

def GenericOOOModel : SchedMachineModel {
  int IssueWidth = 4;
  int MicroOpBufferSize = 128;
  int LoadLatency = 4;
  int MispredictPenalty = 8;
  let CompleteModel = 0;
}

let SchedModel = GenericOOOModel in {
def GenericOOODIV : ProcResource<1>;
def GenericOOOIXU : ProcResource<3>;
def GenericOOOALU : ProcResGroup<[GenericOOODIV, GenericOOOIXU]>;
def GenericOOOLSU : ProcResource<2>;
def GenericOOOFPU : ProcResource<2>;
// TODO: Add vector scheduling.
// def GenericOOOVXU : ProcResource<1>;

def : WriteRes<WriteIALU, [GenericOOOALU]>;
def : WriteRes<WriteIALU32, [GenericOOOALU]>;
def : WriteRes<WriteShiftImm, [GenericOOOALU]>;
def : WriteRes<WriteShiftImm32, [GenericOOOALU]>;
def : WriteRes<WriteShiftReg, [GenericOOOALU]>;
def : WriteRes<WriteShiftReg32, [GenericOOOALU]>;

// zba
def : WriteRes<WriteSHXADD, [GenericOOOALU]>;
def : WriteRes<WriteSHXADD32, [GenericOOOALU]>;

// zbb
def : WriteRes<WriteCLZ, [GenericOOOALU]>;
def : WriteRes<WriteCTZ, [GenericOOOALU]>;
def : WriteRes<WriteCPOP, [GenericOOOALU]>;
def : WriteRes<WriteCLZ32, [GenericOOOALU]>;
def : WriteRes<WriteCTZ32, [GenericOOOALU]>;
def : WriteRes<WriteCPOP32, [GenericOOOALU]>;
def : WriteRes<WriteRotateReg, [GenericOOOALU]>;
def : WriteRes<WriteRotateImm, [GenericOOOALU]>;
def : WriteRes<WriteRotateReg32, [GenericOOOALU]>;
def : WriteRes<WriteRotateImm32, [GenericOOOALU]>;
def : WriteRes<WriteREV8, [GenericOOOALU]>;
def : WriteRes<WriteORCB, [GenericOOOALU]>;
def : WriteRes<WriteIMinMax, [GenericOOOALU]>;

let Latency = 0 in
def : WriteRes<WriteNop, [GenericOOOALU]>;

let Latency = 4 in {
def : WriteRes<WriteLDB, [GenericOOOLSU]>;
def : WriteRes<WriteLDH, [GenericOOOLSU]>;
def : WriteRes<WriteLDW, [GenericOOOLSU]>;
def : WriteRes<WriteLDD, [GenericOOOLSU]>;

def : WriteRes<WriteAtomicW, [GenericOOOLSU]>;
def : WriteRes<WriteAtomicD, [GenericOOOLSU]>;
def : WriteRes<WriteAtomicLDW, [GenericOOOLSU]>;
def : WriteRes<WriteAtomicLDD, [GenericOOOLSU]>;
}

let Latency = 6 in {
def : WriteRes<WriteFLD32, [GenericOOOLSU]>;
def : WriteRes<WriteFLD64, [GenericOOOLSU]>;
}

def : WriteRes<WriteSTB, [GenericOOOLSU]>;
def : WriteRes<WriteSTH, [GenericOOOLSU]>;
def : WriteRes<WriteSTW, [GenericOOOLSU]>;
def : WriteRes<WriteSTD, [GenericOOOLSU]>;

def : WriteRes<WriteAtomicSTW, [GenericOOOLSU]>;
def : WriteRes<WriteAtomicSTD, [GenericOOOLSU]>;

def : WriteRes<WriteFST32, [GenericOOOLSU]>;
def : WriteRes<WriteFST64, [GenericOOOLSU]>;

let Latency = 6 in {
def : WriteRes<WriteFMovI32ToF32, [GenericOOOLSU]>;
def : WriteRes<WriteFMovF32ToI32, [GenericOOOLSU]>;
def : WriteRes<WriteFMovI64ToF64, [GenericOOOLSU]>;
def : WriteRes<WriteFMovF64ToI64, [GenericOOOLSU]>;
}

let Latency = 4 in {
def : WriteRes<WriteIMul, [GenericOOOALU]>;
def : WriteRes<WriteIMul32, [GenericOOOALU]>;
}

let Latency = 13, ReleaseAtCycles = [13] in {
def : WriteRes<WriteIDiv, [GenericOOODIV]>;
def : WriteRes<WriteIDiv32,  [GenericOOODIV]>;
def : WriteRes<WriteIRem, [GenericOOODIV]>;
def : WriteRes<WriteIRem32, [GenericOOODIV]>;
}

def : WriteRes<WriteCSR, [GenericOOOALU]>;

def : WriteRes<WriteJmp, [GenericOOOALU]>;
def : WriteRes<WriteJalr, [GenericOOOALU]>;
def : WriteRes<WriteJal, [GenericOOOALU]>;
def : WriteRes<WriteJalr, [GenericOOOALU]>;

let Latency = 4 in {
def : WriteRes<WriteFCvtI32ToF32, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtI32ToF64, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtI64ToF32, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtI64ToF64, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF32ToI32, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF32ToI64, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF32ToF64, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF64ToI32, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF64ToI64, [GenericOOOFPU]>;
def : WriteRes<WriteFCvtF64ToF32, [GenericOOOFPU]>;

def : WriteRes<WriteFAdd32, [GenericOOOFPU]>;
def : WriteRes<WriteFAdd64, [GenericOOOFPU]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ32, [GenericOOOFPU]>;
def : WriteRes<WriteFMinMax32, [GenericOOOFPU]>;
def : WriteRes<WriteFSGNJ64, [GenericOOOFPU]>;
def : WriteRes<WriteFMinMax64, [GenericOOOFPU]>;

def : WriteRes<WriteFCmp32, [GenericOOOFPU]>;
def : WriteRes<WriteFCmp64, [GenericOOOFPU]>;
}

def : WriteRes<WriteFClass32, [GenericOOOFPU]>;
def : WriteRes<WriteFClass64, [GenericOOOFPU]>;

let Latency = 6 in {
def : WriteRes<WriteFMA32, [GenericOOOFPU]>;
def : WriteRes<WriteFMA64, [GenericOOOFPU]>;
}

let Latency = 5 in {
def : WriteRes<WriteFMul32, [GenericOOOFPU]>;
def : WriteRes<WriteFMul64, [GenericOOOFPU]>;
}

let Latency = 13, ReleaseAtCycles = [13] in {
def : WriteRes<WriteFDiv32, [GenericOOOFPU]>;
def : WriteRes<WriteFSqrt32, [GenericOOOFPU]>;
}

let Latency = 17, ReleaseAtCycles = [17] in {
def : WriteRes<WriteFDiv64, [GenericOOOFPU]>;
def : WriteRes<WriteFSqrt64, [GenericOOOFPU]>;
}

// Bypass and advance.
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;

// Unsupported extensions.
defm : UnsupportedSchedV;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedZvkned;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsfvcp;

defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfhmin;
defm : UnsupportedSchedZabha;
}
