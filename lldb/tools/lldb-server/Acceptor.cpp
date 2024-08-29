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

#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UriParser.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace llvm;

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

  if (m_named_socket) {
    conn = ConnNamed;
    return m_named_socket->Accept(conn_socket);
  }

  if (m_listen_sockets.size() == 0) {
    return Status::FromErrorString("No open listening sockets!");
  }

  Status error;
  NativeSocket accepted_fd = Socket::kInvalidSocketValue;
  NativeSocket listen_fd = Socket::kInvalidSocketValue;
  lldb_private::SocketAddress accept_addr;
  std::vector<MainLoopBase::ReadHandleUP> handles;
  for (auto socket : m_listen_sockets) {
    NativeSocket fd = socket.first;
    IOObjectSP io_sp = IOObjectSP(new TCPSocket(
        fd, /*should_close=*/false, /*child_processes_inherit=*/false));
    handles.emplace_back(m_accept_loop.RegisterReadObject(
        io_sp,
        [fd, &accepted_fd, &accept_addr, &error,
         &listen_fd](MainLoopBase &loop) {
          socklen_t sa_len = accept_addr.GetMaxLength();
          accepted_fd =
              Socket::AcceptSocket(fd, &accept_addr.sockaddr(), &sa_len,
                                   /*child_processes_inherit=*/false, error);
          listen_fd = fd;
          loop.RequestTermination();
        },
        error));
    if (error.Fail())
      return error;
  }

  for (;;) {
    m_accept_loop.Run();
    if (m_break_loop)
      error = Status::FromErrorString("Exiting requested");
    if (error.Fail())
      return error;

    lldb_private::SocketAddress &addr_in = m_listen_sockets[listen_fd];
    if (!addr_in.IsAnyAddr() && accept_addr != addr_in) {
      if (accepted_fd != Socket::kInvalidSocketValue) {
        Socket::CloseSocket(accepted_fd);
        accepted_fd = Socket::kInvalidSocketValue;
      }
      llvm::errs() << llvm::formatv(
          "error: rejecting incoming connection from {0} (expecting {1})\n",
          accept_addr.GetIPAddress(), addr_in.GetIPAddress());
      continue;
    }
    uint16_t local_port = 0;
    socklen_t sa_len = addr_in.GetLength();
    if (::getsockname(accepted_fd, &addr_in.sockaddr(), &sa_len) == 0)
      local_port = addr_in.GetPort();
    if (local_port == m_platform_port) {
      conn = ConnTCPPlatform;
    } else if (local_port == m_gdbserver_port) {
      conn = ConnTCPGdbServer;
    } else {
      if (accepted_fd != Socket::kInvalidSocketValue) {
        Socket::CloseSocket(accepted_fd);
        accepted_fd = Socket::kInvalidSocketValue;
      }
      llvm::errs() << llvm::formatv(
          "error: rejecting incoming connection to port {0}\n", local_port);
      continue;
    }

    break;
  }
  TCPSocket *accepted_socket = new TCPSocket(accepted_fd, /*should_close=*/true,
                                             /*child_processes_inherit=*/false);
  // Keep our TCP packets coming without any delays.
  accepted_socket->SetOptionNoDelay();
  conn_socket = accepted_socket;
  return Status();
}

void Acceptor::BreakAccept() {
  m_break_loop = true;
  m_accept_loop.AddPendingCallback(
      [](MainLoopBase &loop) { loop.RequestTermination(); });
}

Acceptor::Acceptor(StringRef name, uint16_t gdbserver_port, int backlog,
                   Status &error)
    : m_platform_port(0), m_gdbserver_port(gdbserver_port),
      m_break_loop(false) {
  error.Clear();

  m_socket_protocol = Socket::ProtocolUnixDomain;
  std::string hostname;
  // Try to match socket name as URL - e.g., tcp://localhost:5555
  if (std::optional<URI> res = URI::Parse(name)) {
    if (!Socket::FindProtocolByScheme(res->scheme.str().c_str(),
                                      m_socket_protocol)) {
      error = Status::FromErrorStringWithFormat(
          "Unknown protocol scheme \"%s\"", res->scheme.str().c_str());
      return;
    }
    name = name.drop_front(res->scheme.size() + strlen("://"));
    if (m_socket_protocol == Socket::ProtocolTcp) {
      if (res->port) {
        m_platform_port = *(res->port);
      }
      hostname = res->hostname;
    }
  } else {
    // Try to match socket name as $host:port - e.g., localhost:5555
    llvm::Expected<Socket::HostAndPort> host_port =
        Socket::DecodeHostAndPort(name);
    if (!llvm::errorToBool(host_port.takeError())) {
      m_socket_protocol = Socket::ProtocolTcp;
      m_platform_port = host_port->port;
      hostname = host_port->hostname;
    }
  }

  m_name = name.str();
  if (m_socket_protocol != Socket::ProtocolTcp) {
    m_named_socket = Socket::Create(m_socket_protocol,
                                    /*child_processes_inherit=*/false, error);
    error = m_named_socket->Listen(name, backlog);
    return;
  }

  if (m_platform_port != 0 && m_platform_port == m_gdbserver_port) {
    error = Status::FromErrorStringWithFormat(
        "The same ports \"%s\" and %u", name.str().c_str(), m_gdbserver_port);
    return;
  }

  if (hostname == "*")
    hostname = "0.0.0.0";

  uint16_t *ports[] = {&m_platform_port, &m_gdbserver_port};

  std::vector<SocketAddress> addresses = SocketAddress::GetAddressInfo(
      hostname.c_str(), nullptr, AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP);
  for (SocketAddress &address : addresses) {
    for (size_t i = 0; i < 2; ++i) {
      NativeSocket fd =
          Socket::CreateSocket(address.GetFamily(), SOCK_STREAM, IPPROTO_TCP,
                               /*child_processes_inherit=*/false, error);
      if (error.Fail() || fd < 0)
        continue;

      // enable local address reuse
      int option_value = 1;
      set_socket_option_arg_type option_value_p =
          reinterpret_cast<set_socket_option_arg_type>(&option_value);
      if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, option_value_p,
                       sizeof(option_value)) == -1) {
        Socket::CloseSocket(fd);
        continue;
      }

      SocketAddress listen_address = address;
      if (!listen_address.IsLocalhost())
        listen_address.SetToAnyAddress(address.GetFamily(), *ports[i]);
      else
        listen_address.SetPort(*ports[i]);

      int err =
          ::bind(fd, &listen_address.sockaddr(), listen_address.GetLength());
      if (err != -1)
        err = ::listen(fd, backlog);

      if (err == -1) {
        error = Socket::GetLastError();
        Socket::CloseSocket(fd);
        continue;
      }

      if (*ports[i] == 0) {
        socklen_t sa_len = address.GetLength();
        if (::getsockname(fd, &address.sockaddr(), &sa_len) == 0)
          *ports[i] = address.GetPort();
      }

      m_listen_sockets[fd] = address;
    }
  }

  if (error.Success()) {
    assert(m_listen_sockets.size() >= 2);
    assert(m_platform_port);
    assert(m_gdbserver_port);
    assert(m_platform_port != m_gdbserver_port);
  }
}
