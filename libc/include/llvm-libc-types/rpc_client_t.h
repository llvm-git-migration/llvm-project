//===-- Definition of type rpc_port_t -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_RPC_PORT_T_H__
#define __LLVM_LIBC_TYPES_RPC_PORT_T_H__

typedef struct {
  __UINT8_TYPE__ reserved[32];
} rpc_port_t;

typedef struct {
  __UINT64_TYPE__ data[8];
} rpc_buffer_t;

typedef void (*rpc_callback_t)(rpc_buffer_t *buffer, void *data,
                               __UINT32_TYPE__ id);

#endif // __LLVM_LIBC_TYPES_RPC_PORT_T_H__
