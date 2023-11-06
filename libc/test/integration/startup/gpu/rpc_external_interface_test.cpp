//===-- Loader test to check the external RPC interface with the loader ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gpu/rpc.h>

#include "src/gpu/rpc_close_port.h"
#include "src/gpu/rpc_open_port.h"
#include "src/gpu/rpc_recv.h"
#include "src/gpu/rpc_send.h"

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

// Test to ensure that we can use aribtrary combinations of sends and recieves
// as long as they are mirrored using the external interface.
static void test_interface(bool end_with_send) {
  uint64_t cnt = 0;
  // rpc::Client::Port port = rpc::client.open<RPC_TEST_INTERFACE>();
  rpc_port_t port = LIBC_NAMESPACE::rpc_open_port(RPC_TEST_INTERFACE);

  LIBC_NAMESPACE::rpc_send(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        buffer->data[0] = *reinterpret_cast<bool *>(data);
      },
      &end_with_send);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  LIBC_NAMESPACE::rpc_send(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        buffer->data[0] = *reinterpret_cast<uint64_t *>(data) += 1;
      },
      &cnt);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
      },
      &cnt);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  LIBC_NAMESPACE::rpc_send(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        buffer->data[0] = *reinterpret_cast<uint64_t *>(data) += 1;
      },
      &cnt);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
      },
      &cnt);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  LIBC_NAMESPACE::rpc_send(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        buffer->data[0] = *reinterpret_cast<uint64_t *>(data) += 1;
      },
      &cnt);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  LIBC_NAMESPACE::rpc_send(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        buffer->data[0] = *reinterpret_cast<uint64_t *>(data) += 1;
      },
      &cnt);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
      },
      &cnt);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(
      &port,
      [](rpc_buffer_t *buffer, void *data, uint32_t) {
        *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
      },
      &cnt);

  if (end_with_send) {
    // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
    LIBC_NAMESPACE::rpc_send(
        &port,
        [](rpc_buffer_t *buffer, void *data, uint32_t) {
          buffer->data[0] = *reinterpret_cast<uint64_t *>(data) += 1;
        },
        &cnt);
  } else {
    // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
    LIBC_NAMESPACE::rpc_recv(
        &port,
        [](rpc_buffer_t *buffer, void *data, uint32_t) {
          *reinterpret_cast<uint64_t *>(data) = buffer->data[0];
        },
        &cnt);
  }

  // port.close();
  LIBC_NAMESPACE::rpc_close_port(&port);

  ASSERT_TRUE(cnt == 9 && "Invalid number of increments");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_interface(true);
  test_interface(false);

  return 0;
}
