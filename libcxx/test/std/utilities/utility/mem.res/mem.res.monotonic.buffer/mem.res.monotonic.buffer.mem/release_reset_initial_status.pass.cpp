// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory_resource>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"

int main(int, char**) {
  {
    // https://cplusplus.github.io/LWG/issue3120
    {
      // when init given a next buffer size, after release(), reset/not change next buffer size from initial state
      constexpr auto expect_next_buffer_size{512ULL};
      std::pmr::monotonic_buffer_resource mr{nullptr, expect_next_buffer_size, std::pmr::new_delete_resource()};

      for (int i = 0; i < 100; ++i) {
        std::ignore = mr.allocate(1);
        mr.release();
        ASSERT_WITH_LIBRARY_INTERNAL_ALLOCATIONS(globalMemCounter.checkLastNewSizeGe(expect_next_buffer_size));
      }
    }
    {
      // Given
      // when init given a buffer, after release(), initial ptr will reset to it's initial state
      constexpr auto buffer_size{512ULL};
      char buffer[buffer_size];
      std::pmr::monotonic_buffer_resource mr{buffer, buffer_size, std::pmr::null_memory_resource()};

      auto expect_mem_start = mr.allocate(1);
      for (int i = 0; i < 100; ++i) {
        mr.release();
        auto ths_mem_start = mr.allocate(1);
        assert(expect_mem_start == ths_mem_start);
      }
    }
  }
}
