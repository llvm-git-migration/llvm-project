//=- RISCVCombine.td - Define RISC-V Combine Rules -----------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def RISCVPreLegalizerCombiner: GICombiner<
  "RISCVPreLegalizerCombinerImpl", [all_combines]> {
}

def RISCVO0PreLegalizerCombiner: GICombiner<
  "RISCVO0PreLegalizerCombinerImpl", [optnone_combines]> {
}

def lower_insert_subvector : GICombineRule<
  (defs root:$root),
  (match (G_INSERT_SUBVECTOR $dst, $src1, $src2, $idx):$root),
  (apply [{ lowerInsertSubvector(*${root}, STI); }])
>;

// Post-legalization combines which should happen at all optimization levels.
// (E.g. ones that facilitate matching for the selector) For example, matching
// pseudos.
def RISCVPostLegalizerLowering
    : GICombiner<"RISCVPostLegalizerLoweringImpl", [lower_insert_subvector]> {
}

// Post-legalization combines which are primarily optimizations.
// TODO: Add more combines.
def RISCVPostLegalizerCombiner
    : GICombiner<"RISCVPostLegalizerCombinerImpl",
                 [redundant_and, identity_combines]> {
}
