//===------------------------ SocketMsgSupport.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/ModuleBuildDaemon/Client.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace llvm;

namespace clang::tooling::cc1modbuildd {

Expected<std::unique_ptr<raw_socket_stream>>
connectToSocket(StringRef SocketPath) {

  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  if (!MaybeClient)
    return std::move(MaybeClient.takeError());

  return std::move(*MaybeClient);
}

llvm::Error readFromSocket(raw_socket_stream &Connection,
                           std::string &BufferConsumer) {

  char Buffer[MAX_BUFFER];
  ssize_t n;

  while ((n = Connection.read(Buffer, MAX_BUFFER)) > 0) {
    BufferConsumer.append(Buffer, n);
    // Read until \n... encountered (last line of YAML document)
    if (BufferConsumer.find("\n...") != std::string::npos)
      break;
  }

  if (n < 0) {
    // TODO: now that I am using raw_socket_stream look into if I still should
    // be using errno
    std::string Msg = "socket read error: " + std::string(strerror(errno));
    return llvm::make_error<StringError>(Msg, inconvertibleErrorCode());
  }
  if (n == 0)
    return llvm::make_error<StringError>("EOF", inconvertibleErrorCode());
  return llvm::Error::success();
}

// TODO: Need to add error handling for a write
void writeToSocket(raw_socket_stream &Socket, llvm::StringRef Buffer) {
  Socket << Buffer;
  Socket.flush();
}

} // namespace  clang::tooling::cc1modbuildd
