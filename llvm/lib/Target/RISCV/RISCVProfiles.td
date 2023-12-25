//===------ RISCVProfiles.td - RISC-V Profiles -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class RISCVProfile<string name, list<SubtargetFeature> features>
  : SubtargetFeature<name, "RISCVProfile", NAME,
                     "RISC-V " # name # " profile",
                     features>;

def RVI20U32 : RISCVProfile<"rvi20u32", [Feature32Bit]>;

def RVI20U64 : RISCVProfile<"rvi20u64", [Feature64Bit]>;

def RVA20U64 : RISCVProfile<"rva20u64", [Feature64Bit,
                                         FeatureStdExtM,
                                         FeatureStdExtA,
                                         FeatureStdExtF,
                                         FeatureStdExtD,
                                         FeatureStdExtC,
                                         FeatureStdExtZicsr]>;

def RVA20S64 : RISCVProfile<"rva20s64", [Feature64Bit,
                                         FeatureStdExtM,
                                         FeatureStdExtA,
                                         FeatureStdExtF,
                                         FeatureStdExtD,
                                         FeatureStdExtC,
                                         FeatureStdExtZicsr,
                                         FeatureStdExtZifencei]>;

def RVA22U64 : RISCVProfile<"rva22u64", [Feature64Bit,
                                         FeatureStdExtM,
                                         FeatureStdExtA,
                                         FeatureStdExtF,
                                         FeatureStdExtD,
                                         FeatureStdExtC,
                                         FeatureStdExtZba,
                                         FeatureStdExtZbb,
                                         FeatureStdExtZbs,
                                         FeatureStdExtZfhmin,
                                         FeatureStdExtZicbom,
                                         FeatureStdExtZicbop,
                                         FeatureStdExtZicboz,
                                         FeatureStdExtZkt,
                                         FeatureStdExtZicsr,
                                         FeatureStdExtZihintpause]>;

def RVA22S64 : RISCVProfile<"rva22s64", [Feature64Bit,
                                         FeatureStdExtM,
                                         FeatureStdExtA,
                                         FeatureStdExtF,
                                         FeatureStdExtD,
                                         FeatureStdExtC,
                                         FeatureStdExtZba,
                                         FeatureStdExtZbb,
                                         FeatureStdExtZbs,
                                         FeatureStdExtZfhmin,
                                         FeatureStdExtZicbom,
                                         FeatureStdExtZicbop,
                                         FeatureStdExtZicboz,
                                         FeatureStdExtZkt,
                                         FeatureStdExtZicsr,
                                         FeatureStdExtZifencei,
                                         FeatureStdExtZihintpause,
                                         FeatureStdExtSvpbmt,
                                         FeatureStdExtSvinval]>;
