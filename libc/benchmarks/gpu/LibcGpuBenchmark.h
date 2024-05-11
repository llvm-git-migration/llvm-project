#ifndef LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H
#define LLVM_LIBC_BENCHMARKS_LIBC_GPU_BENCHMARK_H

#include "benchmarks/gpu/timing/timing.h"

#include "benchmarks/gpu/TestLogger.h"
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
  // for (int i = 0; i < 3; i++) {
  //   uint64_t result = latency(f, args...);
  //   BestGuess = result;
  //   write_to_stderr(cpp::to_string(result));
  //   write_to_stderr(cpp::string_view("\n"));
  // }
  tlog << "Best Guess: " << BestGuess << '\n';
  tlog << "Samples: " << Samples << '\n';
  return BestGuess;
};

uint64_t benchmark_wrapper(const BenchmarkOptions &Options,
                           uint64_t (*WrapperFunc)());

template <typename F, typename Arg>
uint64_t benchmark_macro(const BenchmarkOptions &Options, F f, Arg arg) {
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
      uint64_t result = 0;
      SINGLE_INPUT_OUTPUT_LATENCY(f, arg, &result);
      SampleCycles += result;
      tlog << "Macro: " << result << '\n';
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
  // for (int i = 0; i < 3; i++) {
  //   uint64_t result = latency(f, args...);
  //   BestGuess = result;
  //   write_to_stderr(cpp::to_string(result));
  //   write_to_stderr(cpp::string_view("\n"));
  // }
  tlog << "Macro Best Guess: " << BestGuess << '\n';
  tlog << "Samples: " << Samples << '\n';
  return BestGuess;
};

class Test {
  Test *Next = nullptr;

public:
  virtual ~Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runTests();

protected:
  static void addTest(Test *);

private:
  virtual void Run() = 0;
  virtual const char *getName() const = 0;

  static Test *Start;
  static Test *End;
};

template <typename F> class FunctionBenchmark : public Test {
  F Func;
  const char *Name;

public:
  FunctionBenchmark(F Func, char const *Name) : Func(Func), Name(Name) {
    addTest(this);
  }

private:
  void Run() override {
    BenchmarkOptions Options;
    auto latency = benchmark(Options, Func);
    LIBC_NAMESPACE::libc_gpu_benchmarks::tlog << "FnName: " << Name << '\n';
    LIBC_NAMESPACE::libc_gpu_benchmarks::tlog << "FnBenchmark: " << latency
                                              << '\n';
  }
  const char *getName() const override { return Name; }
};

class WrapperBenchmark : public Test {
  using BenchmarkWrapperFunction = uint64_t (*)();
  BenchmarkWrapperFunction Func;
  const char *Name;

public:
  WrapperBenchmark(BenchmarkWrapperFunction Func, char const *Name)
      : Func(Func), Name(Name) {
    addTest(this);
  }

private:
  void Run() override {
    tlog << "Running wrapper: " << Name << '\n';
    // for (int i = 0; i < 10; i++) {
    //   auto overhead = LIBC_NAMESPACE::overhead();
    //   auto result = Func() - overhead;
    //   tlog << "Result: " << result << '\n';
    //   tlog << "Overhead: " << overhead << '\n';
    // }
    BenchmarkOptions Options;
    auto latency = benchmark_wrapper(Options, Func);
    tlog << "FnName: " << Name << '\n';
    tlog << "FnBenchmark: " << latency << '\n';
  }
  const char *getName() const override { return Name; }
};

} // namespace libc_gpu_benchmarks

} // namespace LIBC_NAMESPACE

// #define BENCHMARK(SuiteName, TestName)                                         \
//   class SuiteName##_##TestName                                                 \
//       : public LIBC_NAMESPACE::libc_gpu_benchmarks::Test {                     \
//   public:                                                                      \
//     SuiteName##_##TestName() { addTest(this); }                                \
//     void Run() override;                                                       \
//     const char *getName() const override { return #SuiteName "." #TestName; }  \
//   };                                                                           \
//   SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
//   void SuiteName##_##TestName::Run()

#define BENCHMARK_SINGLE_INPUT_OUTPUT(SuiteName, TestName, Func, Arg)          \
  class SuiteName##_##TestName                                                 \
      : public LIBC_NAMESPACE::libc_gpu_benchmarks::Test {                     \
  public:                                                                      \
    SuiteName##_##TestName() { addTest(this); }                                \
    void Run() override;                                                       \
    const char *getName() const override { return #SuiteName "." #TestName; }  \
  };                                                                           \
  SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
  void SuiteName##_##TestName::Run() {                                         \
    LIBC_NAMESPACE::libc_gpu_benchmarks::BenchmarkOptions Options;             \
    LIBC_NAMESPACE::libc_gpu_benchmarks::benchmark(Options, &Func, Arg);       \
  }

#define BENCHMARK_FN(SuiteName, TestName, Func)                                \
  LIBC_NAMESPACE::libc_gpu_benchmarks::FunctionBenchmark                       \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#define BENCHMARK_WRAPPER(SuiteName, TestName, Func)                           \
  LIBC_NAMESPACE::libc_gpu_benchmarks::WrapperBenchmark                        \
      SuiteName##_##TestName##_Instance(Func, #SuiteName "." #TestName);

#define BENCHMARK_S_I_O_V2(SuiteName, TestName, Func, Arg)                     \
  class SuiteName##_##TestName                                                 \
      : public LIBC_NAMESPACE::libc_gpu_benchmarks::Test {                     \
  public:                                                                      \
    SuiteName##_##TestName() { addTest(this); }                                \
    void Run() override;                                                       \
    const char *getName() const override { return #SuiteName "." #TestName; }  \
  };                                                                           \
  SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
  void SuiteName##_##TestName::Run() {                                         \
    LIBC_NAMESPACE::libc_gpu_benchmarks::BenchmarkOptions Options;             \
    LIBC_NAMESPACE::libc_gpu_benchmarks::benchmark_macro(Options, &Func, Arg); \
  }

#endif
