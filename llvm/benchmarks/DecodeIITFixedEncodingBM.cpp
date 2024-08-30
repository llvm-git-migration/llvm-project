//===- DecodeIITFixedEncodingBM.cpp - IIT signature encoding benchmark ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include <variant>

using namespace llvm;
using namespace Intrinsic;

#define GET_INTRINSIC_GENERATOR_GLOBAL
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_GENERATOR_GLOBAL

static void BM_DecodeIITFixedEncoding(benchmark::State &state) {
  for (auto _ : state) {
    for (ID ID = 1; ID < num_intrinsics; ++ID)
      decodeIITFixedEncoding(ID);
  }
}

static void BM_GetIntrinsicInfoTableEntries(benchmark::State &state) {
  SmallVector<IITDescriptor> Table;
  for (auto _ : state) {
    for (ID ID = 1; ID < num_intrinsics; ++ID) {
      // This makes sure the vector does not keep growing, as well as after the
      // first iteration does not result in additional allocations.
      Table.clear();
      getIntrinsicInfoTableEntries(ID, Table);
    }
  }
}

BENCHMARK(BM_DecodeIITFixedEncoding);
BENCHMARK(BM_GetIntrinsicInfoTableEntries);

BENCHMARK_MAIN();
