#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/ctype/isalnum.h"

// BENCHMARK_SINGLE_INPUT_OUTPUT(LlvmLibcIsAlNumGpuBenchmark,
//                               IsAlnumSingleInputOutput,
//                               LIBC_NAMESPACE::isalnum, 'c');

// void BM_IsAlnumBasic() { bool isAlpha = LIBC_NAMESPACE::isalnum('c'); }
// BENCHMARK_FN(LlvmLibcIsAlNumGpuBenchmark, IsAlnumC, BM_IsAlnumBasic);

uint64_t BM_IsAlnumWrapper() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK_WRAPPER(LlvmLibcIsAlNumGpuBenchmark, IsAlnumWrapper,
                  BM_IsAlnumWrapper);

uint64_t BM_IsAlnumWithOverhead() {
  char x = 'c';
  return LIBC_NAMESPACE::function_call_latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK_WRAPPER(LlvmLibcIsAlNumGpuBenchmark, IsAlnumWithOverhead,
                  BM_IsAlnumWithOverhead);
