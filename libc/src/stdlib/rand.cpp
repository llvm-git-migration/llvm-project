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

// This multiplier was obtained from Knuth, D.E., "The Art of
// Computer Programming," Vol 2, Seminumerical Algorithms, Third
// Edition, Addison-Wesley, 1998, p. 106 (line 26) & p. 108 */
LLVM_LIBC_FUNCTION(int, rand, (void)) {
  unsigned long orig = rand_next.load(cpp::MemoryOrder::RELAXED);
  for (;;) {
    uint64_t x = orig;
    x = static_cast<unsigned long>(6364136223846793005ULL) * x;
    if (rand_next.compare_exchange_strong(orig, static_cast<unsigned long>(x),
                                          cpp::MemoryOrder::ACQUIRE,
                                          cpp::MemoryOrder::RELAXED))
      return static_cast<int>(x >> 32) & RAND_MAX;
    sleep_briefly();
  }
}

} // namespace LIBC_NAMESPACE
