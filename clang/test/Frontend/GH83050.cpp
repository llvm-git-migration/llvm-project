/// This is the regression test for https://github.com/llvm/llvm-project/issues/83050.
/// This just needs to compile.
// RUN: %clang_cc1 -x c++ -ffixed-point -S %s -o /dev/null -triple=x86_64-linux -ffixed-point
static constexpr unsigned short _Fract SQRT_FIRST_APPROX[12][2] = {
    {0x1.e8p-1uhr, 0x1.0cp-2uhr}, {0x1.bap-1uhr, 0x1.28p-2uhr},
    {0x1.94p-1uhr, 0x1.44p-2uhr}, {0x1.74p-1uhr, 0x1.6p-2uhr},
    {0x1.6p-1uhr, 0x1.74p-2uhr},  {0x1.4ep-1uhr, 0x1.88p-2uhr},
    {0x1.3ep-1uhr, 0x1.9cp-2uhr}, {0x1.32p-1uhr, 0x1.acp-2uhr},
    {0x1.22p-1uhr, 0x1.c4p-2uhr}, {0x1.18p-1uhr, 0x1.d4p-2uhr},
    {0x1.08p-1uhr, 0x1.fp-2uhr},  {0x1.04p-1uhr, 0x1.f8p-2uhr},
};
