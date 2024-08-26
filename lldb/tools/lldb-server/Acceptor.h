//===-- Acceptor.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H
#define LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H

#include "lldb/Host/Socket.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/Status.h"

#include <functional>
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

  virtual ~Acceptor() = default;

  Status Listen(int backlog);

  Status Accept(Socket *&conn_socket, AcceptorConn &conn);

  Acceptor(llvm::StringRef name, uint16_t gdbserver_port,
           const bool child_processes_inherit, Status &error);

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
  std::unique_ptr<Socket> m_listener_socket_up;
  std::string m_name;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_ACCEPTOR_H
