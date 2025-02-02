//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// There is no way to limit the number of threads on windows
// UNSUPPORTED: msvc

#include <cassert>
#include <future>
#include <system_error>

#if __has_include(<sys/resource.h>)
#  include <sys/resource.h>
#  ifdef RLIMIT_NPROC
void force_thread_creation_failure() {
  rlimit lim = {1, 1};
  setrlimit(RLIMIT_NPROC, &lim);
}
#  else
#    error "No known way to force only one thread being available"
#  endif
#else
#  error "No known way to force only one thread being available"
#endif

int main() {
  force_thread_creation_failure();

  try {
    auto fut = std::async(std::launch::async, [] { return 1; });
    assert(false);
  } catch (const std::system_error&) {
  }

  try {
    auto fut = std::async(std::launch::async, [] { return; });
    assert(false);
  } catch (const std::system_error&) {
  }
}
