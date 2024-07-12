//===-- Implementation of rand --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand.h"
#include "src/__support/common.h"
#include "src/__support/threads/sleep.h"
#include "src/stdlib/rand_util.h"

namespace LIBC_NAMESPACE {

// An implementation of the xorshift* pseudo random number generator. This
// is a good general purpose generator for most non-cryptographics applications.
static inline unsigned long xorshiftstar(unsigned long a, unsigned long b,
                                         unsigned long c, unsigned long d) {
  unsigned long orig = rand_next.load(cpp::MemoryOrder::RELAXED);
  for (;;) {
    unsigned long x = orig;
    x ^= x >> a;
    x ^= x << b;
    x ^= x >> c;
    if (rand_next.compare_exchange_strong(orig, x, cpp::MemoryOrder::ACQUIRE,
                                          cpp::MemoryOrder::RELAXED))
      return x * d;
    sleep_briefly();
  }
}

// An implementation of the xorshift64star pseudo random number generator. This
// is a good general purpose generator for most non-cryptographics applications.
LLVM_LIBC_FUNCTION(int, rand, (void)) {
  int res;
  if constexpr (sizeof(void *) == sizeof(uint64_t))
    res =
        static_cast<int>(xorshiftstar(12, 25, 27, 0x2545F4914F6CDD1Dul) >> 32);
  else
    res = static_cast<int>(xorshiftstar(13, 17, 5, 1597334677ul));
  return res & RAND_MAX;
}

} // namespace LIBC_NAMESPACE
