//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The fix for issue 57964 requires an updated dylib due to explicit
// instantiations. That means Apple backdeployment targets remain broken.
// UNSUPPORTED: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{.+}}
// UNSUPPORTED: stdlib=apple-libc++ && target={{.+}}-apple-macosx11.{{.+}}

// <ios>

// class ios_base

// ~ios_base()
//
// Destroying a constructed ios_base object that has not been
// initialized by basic_ios::init is undefined behaviour. This can
// happen in practice, make sure the undefined behaviour is handled
// gracefully. See ios_base::ios_base() for the details.

#include <ostream>

struct AlwaysThrows {
  AlwaysThrows() { throw 1; }
};

struct Foo : AlwaysThrows, std::ostream {
  Foo() : AlwaysThrows(), std::ostream(nullptr) {}
};

int main() {
  try {
    Foo foo;
  } catch (...) {
  };
  return 0;
}
