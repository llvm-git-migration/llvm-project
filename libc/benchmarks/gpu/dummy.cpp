#include "LibcGpuBenchmark.h"
#include "timing/timing.h"

int add_test(int x) { return x + 1; }

__attribute__((noinline)) [[gnu::noinline]] int function_call_overhead(int x) {
  asm volatile("");
  return x;
}

uint64_t DummyWrapperBenchmark() {
  int x = 10;
  return LIBC_NAMESPACE::latency(add_test, x);
}
BENCHMARK_WRAPPER(Dummy, DummyWrapperBenchmark, DummyWrapperBenchmark);

uint64_t DummyFunctionCallOverhead() {
  int x = 10;
  return LIBC_NAMESPACE::latency(function_call_overhead, x);
}
BENCHMARK_WRAPPER(Dummy, DummyFunctionCallOverhead, DummyFunctionCallOverhead);
