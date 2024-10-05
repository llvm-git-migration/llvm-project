#include <stddef.h>

#if __ARM_FP == 0
// WARNING: When building the scalar versions of these functions you need to
// use the compiler flag "-mllvm -disable-loop-idiom-all" to prevent clang
// from recognising a loop idiom and planting calls to memcpy!

void *__arm_sc_memset(void *dest, int c, size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    destp[i] = c8;

  return dest;
}
#endif

const void *__arm_sc_memchr(const void *src, int c,
                            size_t n) __arm_streaming_compatible {
  const unsigned char *srcp = (const unsigned char *)src;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    if (srcp[i] == c8)
      return &srcp[i];

  return NULL;
}
