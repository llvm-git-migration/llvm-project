//===-- Implementation header for RPC functions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_GPU_RPC_GET_BUFFER_H
#define LLVM_LIBC_SRC_GPU_RPC_GET_BUFFER_H

#include <gpu/rpc.h>

namespace LIBC_NAMESPACE {

rpc_buffer_t *rpc_get_buffer(rpc_port_t *handle);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_GPU_RPC_GET_BUFFER_H
