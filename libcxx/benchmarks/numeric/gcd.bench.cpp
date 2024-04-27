//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <benchmark/benchmark.h>
#include <cstring>
#include <numeric>
#include <random>

template <class T>
static std::array<T, 1000> generate(std::uniform_int_distribution<T> distribution = std::uniform_int_distribution<T>{
                                        std::numeric_limits<T>::min(), std::numeric_limits<T>::max()}) {
  std::mt19937 generator;
  std::array<T, 1000> result;
  std::generate_n(result.begin(), result.size(), [&] { return distribution(generator); });
  return result;
}

static void bm_gcd(benchmark::State& state) {
  std::array data = generate<int>();
  while (state.KeepRunningBatch(data.size()))
    for (auto v0 : data)
      for (auto v1 : data)
        benchmark::DoNotOptimize(std::gcd(v0, v1));
}
BENCHMARK(bm_gcd);

BENCHMARK_MAIN();
