#include "LibcGpuBenchmark.h"
#include "timing/timing.h"

#include "src/stdio/fputs.h"

int add_test(int x) {
    return x + 1;
}

__attribute__((noinline)) [[gnu::noinline]] int function_call_overhead(int x) {
    asm volatile ("");
    return x;
}

// void DummyHiBenchmark() {
//     LIBC_NAMESPACE::fputs("Hi\n", stderr);
// }
// BENCHMARK_FN(Dummy, DummyHiBenchmark, DummyHiBenchmark);

// void DummyV2Benchmark() {
//     int result = dummy_hi(10);
//     asm volatile("" :: "r"(result));
//     auto test_cycles = LIBC_NAMESPACE::latency(dummy_hi, 10);
//     LIBC_NAMESPACE::libc_gpu_benchmarks::tlog << "In func: " << test_cycles << '\n';
// }
// BENCHMARK_FN(Dummy, DummyV2Benchmark, DummyV2Benchmark);

// BENCHMARK_SINGLE_INPUT_OUTPUT(Dummy, DummySingleInputOutput, dummy_hi, 10);

// BENCHMARK_S_I_O_V2(Dummy, DummySIOMacro, dummy_hi, 10);

// uint64_t DummyWrapperBenchmark() {
//     int x = 10;
//     return LIBC_NAMESPACE::latency(add_test, x);
// }
// BENCHMARK_WRAPPER(Dummy, DummyWrapperBenchmark, DummyWrapperBenchmark);

uint64_t DummyFunctionCallOverhead() {
    int x = 10;
    return LIBC_NAMESPACE::latency(function_call_overhead, x);
}
BENCHMARK_WRAPPER(Dummy, DummyFunctionCallOverhead, DummyFunctionCallOverhead);
