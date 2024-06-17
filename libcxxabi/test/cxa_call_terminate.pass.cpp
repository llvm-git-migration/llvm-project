//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdlib>
#include <exception>

void func() noexcept {
  try {
    throw 1;
  } catch (float) {
  }
}

[[noreturn]] void terminate_handler() {
  assert(std::current_exception() != nullptr);
  std::exit(0);
}

int main(int, char**) {
  std::set_terminate(terminate_handler);
  func();
  assert(false);
}
