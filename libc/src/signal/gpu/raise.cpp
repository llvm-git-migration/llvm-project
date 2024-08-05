//===-- GPU implementation of signal --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"

#include "hdr/types/sigset_t.h"
#include "llvm-libc-types/rpc_opcodes_t.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, raise, (int sig)) {
  // We want to first make sure the server is listening before we exit.
  rpc::Client::Port port = rpc::client.open<RPC_RAISE>();
  int ret;
  port.send_and_recv(
      [=](rpc::Buffer *buf) { buf->data[0] = static_cast<uint64_t>(sig); },
      [&](rpc::Buffer *buf) { ret = static_cast<int>(buf->data[0]); });
  port.close();
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
