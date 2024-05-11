//===------------- NVPTX implementation of timing utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
#define LLVM_LIBC_UTILS_GPU_TIMING_NVPTX

#include "src/__support/GPU/utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

// Returns the overhead associated with calling the profiling region. This
// allows us to substract the constant-time overhead from the latency to
// obtain a true result. This can vary with system load.
[[gnu::noinline]] static uint64_t overhead() {
  volatile uint32_t x = 1;
  uint32_t y = x;
  gpu::sync_threads();
  uint64_t start = gpu::processor_clock();
  asm volatile("" ::"r"(y), "llr"(start));
  uint32_t result = y;
  asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);
  uint64_t stop = gpu::processor_clock();
  gpu::sync_threads();
  volatile auto storage = result;
  return stop - start;
}

// Stimulate a simple function and obtain its latency in clock cycles on the
// system. This function cannot be inlined or else it will disturb the very
// deliccate balance of hard-coded dependencies.
//
// FIXME: This does not work in general on NVPTX because of further
// optimizations ptxas performs. The only way to get consistent results is to
// pass and extra "SHELL:-Xcuda-ptxas -O0" to CMake's compiler flag. This
// negatively implacts performance but it is at least stable.
template <typename F, typename T>
[[gnu::noinline]] static LIBC_INLINE uint64_t latency(F f, T t) {
  // We need to store the input somewhere to guarantee that the compiler will
  // not constant propagate it and remove the profiling region.
  volatile T storage = t;
  T arg = storage;
  asm volatile("" ::"r"(arg));

  // Get the current timestamp from the clock.
  gpu::sync_threads();
  __nvvm_membar_sys();
  uint64_t start = gpu::processor_clock();

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm volatile("" ::"r"(arg), "llr"(start));

  // Run the function under test and return its value.
  auto result = f(arg);

  // This inline assembly performs a no-op which forces the result to both be
  // used and prevents us from exiting this region before it's complete.
  asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  __nvvm_membar_sys();
  gpu::sync_threads();
  asm volatile("" ::"r"(stop));
  volatile T output = result;

  // Return the time elapsed.
  return stop - start;
}

[[gnu::noinline]] static uint64_t single_input_function(int x) {
  asm volatile("" :: "r"(x)); // prevent the compiler from optimizing out x
  return x;
}

template <typename F, typename T>
static LIBC_INLINE uint64_t function_call_latency(F f, T t) {
  auto function_call_overhead = latency(single_input_function, 0);
  return latency(f, t) - function_call_overhead;
}

static LIBC_INLINE uint64_t latency(void (*f)()) {
  // Get the current timestamp from the clock.
  gpu::sync_threads();
  uint64_t start = gpu::processor_clock();

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm volatile("" ::"llr"(start));

  // Run the function under test and return its value.
  f();

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  gpu::sync_threads();

  // Return the time elapsed.
  return stop - start;
}

template <typename F, typename T1, typename T2>
static LIBC_INLINE uint64_t latency(F f, T1 t1, T2 t2) {
  volatile T1 storage = t1;
  volatile T2 storage2 = t2;
  T1 arg = storage;
  T2 arg2 = storage2;
  asm volatile("" ::"r"(arg), "r"(arg2));

  gpu::sync_threads();
  uint64_t start = gpu::processor_clock();

  asm volatile("" ::"r"(arg), "r"(arg2), "llr"(start));

  auto result = f(arg, arg2);

  asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);

  uint64_t stop = gpu::processor_clock();
  gpu::sync_threads();
  asm volatile("" ::"r"(stop));
  volatile auto output = result;

  return stop - start;
}

} // namespace LIBC_NAMESPACE

/**
 * LatencyP must be a pointer to a uint64_t holding the result variable
 */
#define SINGLE_INPUT_OUTPUT_LATENCY(Func, t, LatencyP)                         \
  do {                                                                         \
    tlog << "Latency: " << ((long)(*LatencyP)) << '\n';                        \
    *LatencyP = 200;                                                           \
    volatile auto storage = t;                                                 \
    auto arg = storage;                                                        \
    asm volatile("" ::"r"(arg), "r"(LatencyP));                                \
                                                                               \
    LIBC_NAMESPACE::gpu::sync_threads();                                       \
    uint64_t start = LIBC_NAMESPACE::gpu::processor_clock();                   \
                                                                               \
    asm volatile("" ::"r"(arg), "llr"(start));                                 \
    auto result = Func(arg);                                                   \
    asm volatile("" ::"r"(LatencyP));                                          \
    *LatencyP = 312;                                                           \
    asm volatile("" ::"r"(LatencyP));                                          \
    asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);     \
    asm volatile("" ::"r"(LatencyP));                                          \
    *LatencyP = 499;                                                           \
                                                                               \
    uint64_t stop = gpu::processor_clock();                                    \
    gpu::sync_threads();                                                       \
    volatile auto output = result;                                             \
                                                                               \
    tlog << "Start: " << start << '\n';                                        \
    tlog << "Stop: " << stop << '\n';                                          \
    tlog << "Diff: " << (stop - start) << '\n';                                \
    asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(LatencyP) :);   \
    *LatencyP = stop - start;                                                  \
  } while (0)

#endif // LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
