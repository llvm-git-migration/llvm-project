#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/timing/timing.h"

#include "benchmarks/gpu/BenchmarkLogger.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace libc_gpu_benchmarks {

struct BenchmarkOptions {
  uint32_t InitialIterations = 1;
  uint32_t MaxIterations = 10000000;
  uint32_t MinSamples = 4;
  uint32_t MaxSamples = 1000;
  double Epsilon = 0.01;
  double ScalingFactor = 1.4;
};

struct Measurement {
  size_t Iterations = 0;
  uint64_t ElapsedCycles = 0;
};

class RefinableRuntimeEstimation {
  uint64_t TotalCycles = 0;
  size_t TotalIterations = 0;

public:
  uint64_t Update(const Measurement &M) {
    TotalCycles += M.ElapsedCycles;
    TotalIterations += M.Iterations;
    return TotalCycles / TotalIterations;
  }
};

// Tracks the progression of the runtime estimation
class RuntimeEstimationProgression {
  RefinableRuntimeEstimation RRE;

public:
  uint64_t CurrentEstimation = 0;

  double ComputeImprovement(const Measurement &M) {
    const uint64_t NewEstimation = RRE.Update(M);
    double Ratio = ((double)CurrentEstimation / NewEstimation) - 1.0;

    // Get absolute value
    if (Ratio < 0) {
      Ratio *= -1;
    }

    CurrentEstimation = NewEstimation;
    return Ratio;
  }
};

template <typename F, typename... Args>
uint64_t benchmark(const BenchmarkOptions &Options, F f, Args... args) {
  RuntimeEstimationProgression REP;
  size_t Iterations = Options.InitialIterations;
  if (Iterations < (uint32_t)1) {
    Iterations = 1;
  }
  size_t Samples = 0;
  uint64_t BestGuess = 0;
  uint64_t TotalCycles = 0;
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
  // Nvidia cannot perform LTO, so we need to perform
  // 1 call to "warm up" the function before microbenchmarking
  uint64_t result = latency(f, args...);
  tlog << "Running warm-up iteration: " << result << '\n';
#endif
  for (;;) {
    uint64_t SampleCycles = 0;
    for (uint32_t i = 0; i < Iterations; i++) {
      uint64_t result = latency(f, args...);
      SampleCycles += result;
      tlog << result << '\n';
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
  tlog << "\n";
  return BestGuess;
};

uint64_t benchmark_wrapper(const BenchmarkOptions &Options,
                           uint64_t (*WrapperFunc)());

class Benchmark {
  Benchmark *Next = nullptr;

public:
  virtual ~Benchmark() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runBenchmarks();

protected:
  static void addBenchmark(Benchmark *);

private:
  virtual void Run() = 0;
  virtual const char *getName() const = 0;

  static Benchmark *Start;
  static Benchmark *End;
};

template <typename F> class FunctionBenchmark : public Benchmark {
  F Func;
  const char *Name;

public:
  FunctionBenchmark(F Func, char const *Name) : Func(Func), Name(Name) {
    addBenchmark(this);
  }

private:
  void Run() override {
    BenchmarkOptions Options;
    auto latency = benchmark(Options, Func);
    tlog << "FnName: " << Name << '\n';
    tlog << "FnBenchmark: " << latency << '\n';
    tlog << "\n";
  }
  const char *getName() const override { return Name; }
};

class WrapperBenchmark : public Benchmark {
  using BenchmarkWrapperFunction = uint64_t (*)();
  BenchmarkWrapperFunction Func;
  const char *Name;

public:
  WrapperBenchmark(BenchmarkWrapperFunction Func, char const *Name)
      : Func(Func), Name(Name) {
    addBenchmark(this);
  }

private:
  void Run() override {
    tlog << "Running wrapper: " << Name << '\n';
    BenchmarkOptions Options;
    auto latency = benchmark_wrapper(Options, Func);
    tlog << "FnName: " << Name << '\n';
    tlog << "FnBenchmark: " << latency << '\n';
    tlog << "\n";
  }
  const char *getName() const override { return Name; }
};

} // namespace libc_gpu_benchmarks

} // namespace LIBC_NAMESPACE

#define BENCHMARK_FN(SuiteName, TestName, Func)                                \
  LIBC_NAMESPACE::libc_gpu_benchmarks::FunctionBenchmark                       \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#define BENCHMARK_WRAPPER(SuiteName, TestName, Func)                           \
  LIBC_NAMESPACE::libc_gpu_benchmarks::WrapperBenchmark                        \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#endif
