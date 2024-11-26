// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_extendhfxf2

#include <limits.h>
#include <math.h> // for isnan, isinf
#include <stdio.h>

#include "fp_test.h"

#if __LDBL_MANT_DIG__ == 64 && defined(__x86_64__) && defined(COMPILER_RT_HAS_FLOAT16)

xf_float __extendhfxf2(TYPE_FP16 f);

int test_extendhfxf2(TYPE_FP16 a, xf_float expected) {
  xf_float x = __extendhfxf2(a);
  uint16_t a_rep = toRep16(a);
  int ret = compareResultF80_F80(x, expected);
  if (ret) {
    printf("error in test__extendhfxf2(%#.4x) = %.20Lf, "
           "expected %.20Lf\n",
           a_rep, x, expected);
  }
  return ret;
}

int main() {
  // Small positive value
  if (test_extendhfxf2(fromRep16(0x2e66), 0.09997558593750000000L))
    return 1;

  // Small negative value
  if (test_extendhfxf2(fromRep16(0xae66), -0.09997558593750000000L))
    return 1;

  // Zero
  if (test_extendhfxf2(fromRep16(0), 0.0L))
    return 1;

  // Smallest positive non-zero value
  if (test_extendhfxf2(fromRep16(0x0100), 0x1p-16L))
    return 1;

  // Smallest negative non-zero value
  if (test_extendhfxf2(fromRep16(0x8100), -0x1p-16L))
    return 1;

  // Positive infinity
  if (test_extendhfxf2(makeInf16(), makeInf80()))
    return 1;

  // Negative infinity
  if (test_extendhfxf2(makeNegativeInf16(), makeNegativeInf80()))
    return 1;

  // NaN
  if (test_extendhfxf2(makeQNaN16(), makeQNaN80()))
    return 1;

  return 0;
}

#else

int main() {
  printf("skipped\n");
  return 0;
}

#endif
