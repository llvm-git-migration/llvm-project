//===-- llvm/Support/raw_socket_stream.h - Socket streams --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to communicate
// via UNIX sockets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_SOCKET_STREAM_H
#define LLVM_SUPPORT_RAW_SOCKET_STREAM_H

#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>

namespace llvm {

class raw_socket_stream;

#ifdef _WIN32
/// @brief Ensures proper initialization and cleanup of winsock resources
///
/// @details
/// Make sure that calls to WSAStartup and WSACleanup are balanced.
class WSABalancer {
public:
  WSABalancer();
  ~WSABalancer();
};
#endif // _WIN32

/// @class ListeningSocket
/// @brief Manages a passive (i.e., listening) UNIX domain socket
///
/// The ListeningSocket class encapsulates a UNIX domain socket that can listen
/// and accept incoming connections. ListeningSocket is portable and supports
/// Windows builds begining with Insider Build 17063. ListeningSocket is
/// designed for server-side operations, working alongside raw_socket_streams
/// that function as client connections.
///
/// Usage example:
/// @code{.cpp}
/// std::string Path = "/path/to/socket"
/// Expected<ListeningSocket> S = ListeningSocket::createListeningSocket(Path);
///
/// if (listeningSocket) {
///     auto connection = S->accept();
///     if (connection) {
///         // Use the accepted raw_socket_stream for communication.
///     }
/// }
/// @endcode
///
class ListeningSocket {
  std::atomic<int> FD;
  std::string SocketPath;
  ListeningSocket(int SocketFD, StringRef SocketPath);

#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  ~ListeningSocket();
  ListeningSocket(ListeningSocket &&LS);
  ListeningSocket(const ListeningSocket &LS) = delete;
  ListeningSocket &operator=(const ListeningSocket &) = delete;

  /// Closes the socket's FD and unlinks the socket file from the file system.
  /// The method is idempotent
  void shutdown();

  /// Accepts an incoming connection on the listening socket. This method can
  /// optionally either block until a connection is available or timeout after a
  /// specified amount of time has passed. By default the method will block
  /// until the socket has recieved a connection
  ///
  /// @param Timeout An optional timeout duration in microseconds
  ///
  Expected<std::unique_ptr<raw_socket_stream>>
  accept(std::optional<std::chrono::microseconds> Timeout = std::nullopt);

  /// Creates a listening socket bound to the specified file system path.
  /// Handles the socket creation, binding, and immediately starts listening for
  /// incoming connections.
  ///
  /// @param SocketPath The file system path where the socket will be created
  /// @param MaxBacklog The max number of connections in a socket's backlog
  ///
  static Expected<ListeningSocket> createListeningSocket(
      StringRef SocketPath,
      int MaxBacklog = llvm::hardware_concurrency().compute_thread_count());
};

//===----------------------------------------------------------------------===//
//  raw_socket_stream
//===----------------------------------------------------------------------===//

class raw_socket_stream : public raw_fd_stream {
  uint64_t current_pos() const override { return 0; }
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  // TODO: Should probably be private
  raw_socket_stream(int SocketFD);
  /// Create a \p raw_socket_stream connected to the Unix domain socket at \p
  /// SocketPath.
  static Expected<std::unique_ptr<raw_socket_stream>>
  createConnectedSocket(StringRef SocketPath);
  ~raw_socket_stream();
};

} // end namespace llvm

#endif
