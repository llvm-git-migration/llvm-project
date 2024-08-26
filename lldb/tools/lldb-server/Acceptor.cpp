//===-- Acceptor.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Acceptor.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"

#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UriParser.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace llvm;

Status Acceptor::Listen(int backlog) {
  if (m_socket_protocol == Socket::ProtocolTcp) {
    // Update the url with the second port to listen on.
    m_name += llvm::formatv(",{0}", m_gdbserver_port);
  }

  Status error = m_listener_socket_up->Listen(StringRef(m_name), backlog);
  if (error.Fail())
    return error;

  if (m_socket_protocol == Socket::ProtocolTcp) {
    if (m_platform_port == 0 || m_gdbserver_port == 0) {
      TCPSocket *tcp_socket =
          static_cast<TCPSocket *>(m_listener_socket_up.get());
      std::set<uint16_t> ports = tcp_socket->GetLocalPortNumbers();
      std::set<uint16_t>::iterator port_i = ports.begin();
      while (m_platform_port == 0 || m_gdbserver_port == 0) {
        if (port_i == ports.end())
          return Status("cannot resolve ports");
        if (m_platform_port == 0 && *port_i != m_gdbserver_port) {
          m_platform_port = *port_i;
        } else if (m_gdbserver_port == 0 && *port_i != m_platform_port) {
          m_gdbserver_port = *port_i;
        }
        ++port_i;
      }
    }
    assert(m_platform_port);
    assert(m_gdbserver_port);
    assert(m_platform_port != m_gdbserver_port);
  }
  return error;
}

std::string Acceptor::GetLocalSocketId() const {
  if (m_socket_protocol == Socket::ProtocolTcp) {
    return (m_platform_port != 0) ? llvm::to_string(m_platform_port) : "";
  } else {
    return m_name;
  }
}

Status Acceptor::Accept(Socket *&conn_socket, AcceptorConn &conn) {
  conn_socket = nullptr;
  conn = ConnUnknown;
  Status error = m_listener_socket_up->Accept(conn_socket);
  if (error.Fail())
    return error;
  if (conn_socket->GetSocketProtocol() == Socket::ProtocolTcp) {
    TCPSocket *tcp_socket = static_cast<TCPSocket *>(conn_socket);
    uint16_t port = tcp_socket->GetLocalPortNumber();
    if (port == m_platform_port)
      conn = ConnTCPPlatform;
    else if (port == m_gdbserver_port)
      conn = ConnTCPGdbServer;
    else {
      delete conn_socket;
      return Status("unexpected TCP socket port");
    }
  } else {
    conn = ConnNamed;
  }
  return Status();
}

Acceptor::Acceptor(StringRef name, uint16_t gdbserver_port,
                   const bool child_processes_inherit, Status &error)
    : m_platform_port(0), m_gdbserver_port(gdbserver_port) {
  error.Clear();

  m_socket_protocol = Socket::ProtocolUnixDomain;
  // Try to match socket name as URL - e.g., tcp://localhost:5555
  if (std::optional<URI> res = URI::Parse(name)) {
    if (!Socket::FindProtocolByScheme(res->scheme.str().c_str(),
                                      m_socket_protocol)) {
      error.SetErrorStringWithFormat("Unknown protocol scheme \"%s\"",
                                     res->scheme.str().c_str());
      return;
    }
    name = name.drop_front(res->scheme.size() + strlen("://"));
    if (m_socket_protocol == Socket::ProtocolTcp && res->port) {
      m_platform_port = *(res->port);
    }
  } else {
    // Try to match socket name as $host:port - e.g., localhost:5555
    llvm::Expected<Socket::HostAndPort> host_port =
        Socket::DecodeHostAndPort(name);
    if (!llvm::errorToBool(host_port.takeError())) {
      m_socket_protocol = Socket::ProtocolTcp;
      m_platform_port = host_port->port;
    }
  }

  if (m_socket_protocol == Socket::ProtocolTcp && m_platform_port != 0 &&
      m_platform_port == m_gdbserver_port) {
    error.SetErrorStringWithFormat("The same ports \"%s\" and %u",
                                   name.str().c_str(), m_gdbserver_port);
    return;
  }

  m_name = name.str();

  m_listener_socket_up =
      Socket::Create(m_socket_protocol, child_processes_inherit, error);
}
