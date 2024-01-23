//===-- llvm/Support/raw_socket_stream.cpp - Socket streams --*- C++ -*-===//
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

#include "llvm/Support/raw_socket_stream.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Error.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#else
#include "llvm/Support/Windows/WindowsSupport.h"
// winsock2.h must be included before afunix.h
// clang-format off
#include <winsock2.h>
#include <afunix.h>
// clang-format on
#include <io.h>
#endif // _WIN32

#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

using namespace llvm;

#ifdef _WIN32
WSABalancer::WSABalancer() {
  WSADATA WsaData;
  ::memset(&WsaData, 0, sizeof(WsaData));
  if (WSAStartup(MAKEWORD(2, 2), &WsaData) != 0) {
    llvm::report_fatal_error("WSAStartup failed");
  }
}

WSABalancer::~WSABalancer() { WSACleanup(); }
#endif // _WIN32

static std::error_code getLastSocketErrorCode() {
#ifdef _WIN32
  return std::error_code(::WSAGetLastError(), std::system_category());
#else
  return std::error_code(errno, std::system_category());
#endif
}

ListeningSocket::ListeningSocket(int SocketFD, StringRef SocketPath)
    : FD(SocketFD), SocketPath(SocketPath) {}

ListeningSocket::ListeningSocket(ListeningSocket &&LS)
    : FD(LS.FD), SocketPath(LS.SocketPath) {
  LS.FD = -1;
  LS.SocketPath.clear();
}

Expected<ListeningSocket> ListeningSocket::createUnix(StringRef SocketPath,
                                                      int MaxBacklog) {

#ifdef _WIN32
  WSABalancer _;
  SOCKET MaybeSocket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeSocket == INVALID_SOCKET) {
#else
  int MaybeSocket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeSocket == -1) {
#endif
    return llvm::make_error<StringError>("Socket create failed: ",
                                         getLastSocketErrorCode());
  }

  struct sockaddr_un Addr;
  ::memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  ::strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);

  if (::bind(MaybeSocket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1) {
    ::close(MaybeSocket);
    return llvm::make_error<StringError>("Bind error: ",
                                         getLastSocketErrorCode());
  }

  if (::listen(MaybeSocket, MaxBacklog) == -1) {
    return llvm::make_error<StringError>("Listen error: ",
                                         getLastSocketErrorCode());
    ::close(MaybeSocket);
  }

  int UnixSocket;
#ifdef _WIN32
  UnixSocket = _open_osfhandle(MaybeSocket, 0);
#else
  UnixSocket = MaybeSocket;
#endif // _WIN32
  return ListeningSocket{UnixSocket, SocketPath};
}

Expected<std::unique_ptr<raw_socket_stream>>
ListeningSocket::accept(bool Block) {

  std::error_code AcceptEC;

#ifdef _WIN32
  SOCKET WinServerSock = _get_osfhandle(FD);
  if (WinServerSock == INVALID_SOCKET)
    return llvm::make_error<StringError>("Failed to get file handle: ",
                                         getLastSocketErrorCode());

  // Set to non-blocking if required
  if (!Block) {
    u_long BlockingMode = 1;
    if (ioctlsocket(WinServerSock, FIONBIO, &BlockingMode) == SOCKET_ERROR)
      return llvm::make_error<StringError>(
          "Failed to set socket to non-blocking: ", getLastSocketErrorCode());
  }

  SOCKET WinAcceptSock = ::accept(WinServerSock, NULL, NULL);
  if (WinAcceptSock == INVALID_SOCKET)
    AcceptEC = getLastSocketErrorCode();

  // Restore to blocking if required
  if (!Block) {
    u_long BlockingMode = 0;
    if (ioctlsocket(WinServerSock, FIONBIO, &BlockingMode) == SOCKET_ERROR)
      return llvm::make_error<StringError>(
          "Failed to reset socket to blocking: ", getLastSocketErrorCode());
  }

  if (WinAcceptSock == INVALID_SOCKET)
    return llvm::make_error<StringError>("Accept Failed: ", AcceptEC);

  int AcceptFD = _open_osfhandle(WinAcceptSock, 0);
  if (AcceptFD == -1)
    return llvm::make_error<StringError>(
        "Failed to get file descriptor from handle: ",
        getLastSocketErrorCode());
#else
  int Flags = ::fcntl(FD, F_GETFL, 0);
  if (Flags == -1)
    return llvm::make_error<StringError>(
        "Failed to get file descriptor flags: ", getLastSocketErrorCode());

  // Set to non-blocking if required
  if (!Block) {
    if (::fcntl(FD, F_SETFL, Flags | O_NONBLOCK) == -1)
      return llvm::make_error<StringError>(
          "Failed to set socket to non-blocking: ", getLastSocketErrorCode());
  }

  int AcceptFD = ::accept(FD, NULL, NULL);
  if (AcceptFD == -1)
    AcceptEC = getLastSocketErrorCode();

  // Restore to blocking if required
  if (!Block) {
    if (::fcntl(FD, F_SETFL, Flags) == -1)
      return llvm::make_error<StringError>(
          "Failed to reset socket to blocking: ", getLastSocketErrorCode());
  }

  if (AcceptFD == -1)
    return llvm::make_error<StringError>("Accept Failed: ", AcceptEC);

#endif //_WIN32

  return std::make_unique<raw_socket_stream>(AcceptFD);
}

ListeningSocket::~ListeningSocket() {
  if (FD == -1)
    return;
  ::close(FD);
  ::unlink(SocketPath.c_str());
}

static Expected<int> GetSocketFD(StringRef SocketPath) {
#ifdef _WIN32
  SOCKET MaybeSocket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeSocket == INVALID_SOCKET) {
#else
  int MaybeSocket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (MaybeSocket == -1) {
#endif // _WIN32
    return llvm::make_error<StringError>("Create socket failed: ",
                                         getLastSocketErrorCode());
  }

  struct sockaddr_un Addr;
  memset(&Addr, 0, sizeof(Addr));
  Addr.sun_family = AF_UNIX;
  strncpy(Addr.sun_path, SocketPath.str().c_str(), sizeof(Addr.sun_path) - 1);

  if (::connect(MaybeSocket, (struct sockaddr *)&Addr, sizeof(Addr)) == -1)
    return llvm::make_error<StringError>("Connect socket failed: ",
                                         getLastSocketErrorCode());

#ifdef _WIN32
  return _open_osfhandle(MaybeSocket, 0);
#else
  return MaybeSocket;
#endif // _WIN32
}

raw_socket_stream::raw_socket_stream(int SocketFD)
    : raw_fd_stream(SocketFD, true) {}

Expected<std::unique_ptr<raw_socket_stream>>
raw_socket_stream::createConnectedUnix(StringRef SocketPath) {
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32
  Expected<int> FD = GetSocketFD(SocketPath);
  if (!FD)
    return FD.takeError();
  return std::make_unique<raw_socket_stream>(*FD);
}

raw_socket_stream::~raw_socket_stream() {}

//===----------------------------------------------------------------------===//
//  raw_string_ostream
//===----------------------------------------------------------------------===//

void raw_string_ostream::write_impl(const char *Ptr, size_t Size) {
  OS.append(Ptr, Size);
}
