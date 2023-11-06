//===-- Loader test to check the external RPC interface with the loader ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gpu/rpc.h>

#include "src/gpu/rpc_close_port.h"
#include "src/gpu/rpc_get_buffer.h"
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

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = end_with_send; });
  rpc_buffer_t *buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  buffer->data[0] = end_with_send;
  LIBC_NAMESPACE::rpc_send(&port);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  buffer->data[0] = cnt = cnt + 1;
  LIBC_NAMESPACE::rpc_send(&port);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(&port);
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  cnt = buffer->data[0];

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  buffer->data[0] = cnt = cnt + 1;
  LIBC_NAMESPACE::rpc_send(&port);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(&port);
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  cnt = buffer->data[0];

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  buffer->data[0] = cnt = cnt + 1;
  LIBC_NAMESPACE::rpc_send(&port);

  // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  buffer->data[0] = cnt = cnt + 1;
  LIBC_NAMESPACE::rpc_send(&port);

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(&port);
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  cnt = buffer->data[0];

  // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  LIBC_NAMESPACE::rpc_recv(&port);
  buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
  cnt = buffer->data[0];

  if (end_with_send) {
    // port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
    buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
    buffer->data[0] = cnt = cnt + 1;
    LIBC_NAMESPACE::rpc_send(&port);
  } else {
    // port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
    LIBC_NAMESPACE::rpc_recv(&port);
    buffer = LIBC_NAMESPACE::rpc_get_buffer(&port);
    cnt = buffer->data[0];
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
