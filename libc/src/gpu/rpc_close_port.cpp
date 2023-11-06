//===---------- GPU implementation of the external RPC port interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/gpu/rpc_close_port.h"

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

static_assert(sizeof(rpc_port_t) == sizeof(rpc::Client::Port), "ABI mismatch");
static_assert(sizeof(size_t) == sizeof(uint64_t), "Size mismatch");
static_assert(LIBC_HAS_BUILTIN(__builtin_bit_cast), "Bitcast not available");

LLVM_LIBC_FUNCTION(void, rpc_close_port, (rpc_port_t * handle)) {
  rpc::Client::Port port = __builtin_bit_cast(rpc::Client::Port, *handle);
  port.close();
  *handle = cpp::bit_cast<rpc_port_t>(port);
}

} // namespace LIBC_NAMESPACE
