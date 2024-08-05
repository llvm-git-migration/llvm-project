//===-- Interface for freelist --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FREELIST2_H
#define LLVM_LIBC_SRC___SUPPORT_FREELIST2_H

namespace LIBC_NAMESPACE_DECL {

class FreeList {
public:
  class Node {
  };

private:
  Node *begin_;
};

}

} // namespace LIBC_NAMESPACE_DECL
  
#endif // LLVM_LIBC_SRC___SUPPORT_FREELIST2_H
