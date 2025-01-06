; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

define ppc_fp128 @__floatditf(i64 %u) nounwind  {
entry:
        %tmp6 = fmul ppc_fp128 f0x00000000000000000000000000000000, f0x000000000000000041F0000000000000
        %tmp78 = trunc i64 %u to i32
        %tmp789 = uitofp i32 %tmp78 to ppc_fp128
        %tmp11 = fadd ppc_fp128 %tmp789, %tmp6
        ret ppc_fp128 %tmp11
}
