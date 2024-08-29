//===-- Acceptor.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H
#define LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/Socket.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>

namespace llvm {
class StringRef;
}

namespace lldb_private {
namespace lldb_server {

class Acceptor {
public:
  enum AcceptorConn {
    ConnUnknown,
    ConnNamed,
    ConnTCPPlatform,
    ConnTCPGdbServer
  };

  Acceptor(llvm::StringRef name, uint16_t gdbserver_port, int backlog,
           Status &error);
  virtual ~Acceptor() = default;

  Status Accept(Socket *&conn_socket, AcceptorConn &conn);
  void BreakAccept();

  Socket::SocketProtocol GetSocketProtocol() const {
    return m_socket_protocol;
  };

  uint16_t GetPlatformPort() const { return m_platform_port; };
  uint16_t GetGdbServerPort() const { return m_gdbserver_port; };

  // Returns either TCP port number as string or domain socket path.
  // Empty string is returned in case of error.
  std::string GetLocalSocketId() const;

private:
  Socket::SocketProtocol m_socket_protocol;
  uint16_t m_platform_port;
  uint16_t m_gdbserver_port;
  std::string m_name;
  std::unique_ptr<Socket> m_named_socket;
  std::map<NativeSocket, SocketAddress> m_listen_sockets;
  MainLoop m_accept_loop;
  std::atomic<bool> m_break_loop;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H
