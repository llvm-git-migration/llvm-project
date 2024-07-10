//===-- Common utility class for differential analysis --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "test/src/math/performance_testing/Timer.h"

#include <fstream>

namespace LIBC_NAMESPACE {
namespace testing {

template <typename T> class NearestIntegerPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

public:
  typedef T Func(T);

  static void runPerfInRange(Func myFunc, Func otherFunc,
                             StorageType startingBit, StorageType endingBit,
                             StorageType step, size_t rounds,
                             std::ofstream &log) {
    auto runner = [=](Func func) {
      volatile T result;
      for (size_t i = 0; i < rounds; i++) {
        for (StorageType bits = startingBit; bits <= endingBit; bits += step) {
          T x = FPBits(bits).get_val();
          result = func(x);
        }
      }
    };

    Timer timer;
    timer.start();
    runner(myFunc);
    timer.stop();

    size_t numberOfRuns = (endingBit - startingBit) / step + 1;
    double myAverage =
        static_cast<double>(timer.nanoseconds()) / numberOfRuns / rounds;
    log << "-- My function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << myAverage << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / myAverage) << " op/s \n";

    timer.start();
    runner(otherFunc);
    timer.stop();

    double otherAverage =
        static_cast<double>(timer.nanoseconds()) / numberOfRuns / rounds;
    log << "-- Other function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << otherAverage << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / otherAverage) << " op/s \n";

    log << "-- Average runtime ratio --\n";
    log << "     Mine / Other's  : " << myAverage / otherAverage << " \n";
  }

  static void runPerf(Func myFunc, Func otherFunc, size_t rounds,
                      const char *logFile) {
    std::ofstream log(logFile);
    log << "Performance tests with inputs in normal integral range:\n";
    runPerfInRange(myFunc, otherFunc,
                   StorageType((FPBits::EXP_BIAS + 1) << FPBits::SIG_LEN),
                   StorageType((FPBits::EXP_BIAS + FPBits::FRACTION_LEN - 1)
                               << FPBits::SIG_LEN),
                   StorageType(1 << FPBits::SIG_LEN),
                   rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS, log);
    log << "\n Performance tests with inputs in low integral range:\n";
    runPerfInRange(myFunc, otherFunc, StorageType(1 << FPBits::SIG_LEN),
                   StorageType((FPBits::EXP_BIAS - 1) << FPBits::SIG_LEN),
                   StorageType(1 << FPBits::SIG_LEN),
                   rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS, log);
    log << "\n Performance tests with inputs in high integral range:\n";
    runPerfInRange(myFunc, otherFunc,
                   StorageType((FPBits::EXP_BIAS + FPBits::FRACTION_LEN)
                               << FPBits::SIG_LEN),
                   StorageType(FPBits::MAX_BIASED_EXPONENT << FPBits::SIG_LEN),
                   StorageType(1 << FPBits::SIG_LEN),
                   rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS, log);
    log << "\n Performance tests with inputs in normal fractional range:\n";
    runPerfInRange(myFunc, otherFunc,
                   StorageType(((FPBits::EXP_BIAS + 1) << FPBits::SIG_LEN) + 1),
                   StorageType(((FPBits::EXP_BIAS + 2) << FPBits::SIG_LEN) - 1),
                   StorageType(1), rounds * 2, log);
    log << "\n Performance tests with inputs in subnormal fractional range:\n";
    runPerfInRange(myFunc, otherFunc, StorageType(1),
                   StorageType(FPBits::SIG_MASK), StorageType(1), rounds, log);
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define NEAREST_INTEGER_PERF(T, myFunc, otherFunc, rounds, filename)           \
  {                                                                            \
    LIBC_NAMESPACE::testing::NearestIntegerPerf<T>::runPerf(                   \
        &myFunc, &otherFunc, rounds, filename);                                \
    LIBC_NAMESPACE::testing::NearestIntegerPerf<T>::runPerf(                   \
        &myFunc, &otherFunc, rounds, filename);                                \
  }
