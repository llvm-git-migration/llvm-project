
#include "benchmarks/gpu/timing/timing.h"

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"

#include "LibcGpuBenchmark.h"

namespace LIBC_NAMESPACE {
namespace libc_gpu_benchmarks {

Test *Test::Start = nullptr;
Test *Test::End = nullptr;

void Test::addTest(Test *T) {
  if (End == nullptr) {
    Start = T;
    End = T;
    return;
  }

  End->Next = T;
  End = T;
}

int Test::runTests() {
  for (Test *T = Start; T != nullptr; T = T->Next) {
    tlog << T->getName() << "\n";
    T->Run();
  }

  return 0;
}

uint64_t benchmark_wrapper(const BenchmarkOptions &Options,
                           uint64_t (*WrapperFunc)()) {
  RuntimeEstimationProgression REP;
  size_t Iterations = Options.InitialIterations;
  if (Iterations < (uint32_t)1) {
    Iterations = 1;
  }
  size_t Samples = 0;
  uint64_t BestGuess = 0;
  uint64_t TotalCycles = 0;
  for (;;) {
    uint64_t SampleCycles = 0;
    for (uint32_t i = 0; i < Iterations; i++) {
      auto overhead = LIBC_NAMESPACE::overhead();
      uint64_t result = WrapperFunc() - overhead;
      SampleCycles += result;
    }

    Samples++;
    TotalCycles += SampleCycles;
    const double ChangeRatio =
        REP.ComputeImprovement({Iterations, SampleCycles});
    BestGuess = REP.CurrentEstimation;

    if (Samples >= Options.MaxSamples || Iterations >= Options.MaxIterations) {
      break;
    } else if (Samples >= Options.MinSamples && ChangeRatio < Options.Epsilon) {
      tlog << "Samples are stable!\n";
      break;
    }

    Iterations *= Options.ScalingFactor;
  }
  tlog << "Best Guess: " << BestGuess << '\n';
  tlog << "Samples: " << Samples << '\n';
  return BestGuess;
};

} // namespace libc_gpu_benchmarks
} // namespace LIBC_NAMESPACE
