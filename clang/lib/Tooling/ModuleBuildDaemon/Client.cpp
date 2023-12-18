//===----------------------------- Client.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Client.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Program.h"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <optional>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace llvm;

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(raw_socket_stream &Client,
                             DiagnosticsEngine &Diag) {

  HandshakeMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  std::string Buffer = getBufferFromSocketMsg(Request);

  // Send HandshakeMsg to module build daemon
  Client << Buffer;
  Client.flush();

  Buffer.clear();
  // Receive response from module build daemon
  if (llvm::Error ReadErr = readFromSocket(Client, Buffer)) {
    // TODO: Add context such as "Daemon failed to read buffer from socket" to
    // error message
    return std::move(ReadErr);
  }

  Expected<HandshakeMsg> MaybeHandshakeResponse =
      getSocketMsgFromBuffer<HandshakeMsg>(Buffer);
  if (!MaybeHandshakeResponse) {
    // TODO: Add context such as "Failed to convert buffer to HandshakeMsg" to
    // error message
    return std::move(MaybeHandshakeResponse.takeError());
  }

  HandshakeMsg HandshakeResponse = std::move(*MaybeHandshakeResponse);
  assert(HandshakeResponse.MsgAction == ActionType::HANDSHAKE &&
         "Response ActionType should only ever be HANDSHAKE");

  if (HandshakeResponse.MsgStatus == StatusType::SUCCESS) {
    return llvm::Error::success();
  }

  return llvm::make_error<StringError>(
      "Received failed handshake response from module build daemon",
      inconvertibleErrorCode());
}

llvm::Error spawnModuleBuildDaemon(const char *Argv0, DiagnosticsEngine &Diag) {

  std::vector<StringRef> Args = {Argv0, "-cc1modbuildd"};
  std::string ErrorBuffer;
  llvm::sys::ExecuteNoWait(Argv0, Args, std::nullopt, {}, 0, &ErrorBuffer,
                           nullptr, nullptr, /*DetachProcess*/ true);

  if (!ErrorBuffer.empty())
    return llvm::make_error<StringError>(ErrorBuffer, inconvertibleErrorCode());

  Diag.Report(diag::remark_mbd_spawn);
  return llvm::Error::success();
}

Expected<std::unique_ptr<raw_socket_stream>>
getModuleBuildDaemon(const char *Argv0, StringRef BasePath,
                     DiagnosticsEngine &Diag) {

  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
        connectToSocket(SocketPath);
    if (MaybeClient) {
      return std::move(*MaybeClient);
    }
    consumeError(MaybeClient.takeError());
  }

  if (llvm::Error Err = spawnModuleBuildDaemon(Argv0, Diag))
    return std::move(Err);

  const unsigned int MICROSEC_IN_SEC = 1000000;
  const constexpr unsigned int MAX_WAIT_TIME = 30 * MICROSEC_IN_SEC;
  unsigned int CumulativeTime = 0;
  unsigned int WaitTime = 10;

  while (CumulativeTime <= MAX_WAIT_TIME) {
    // Wait a bit then check to see if the module build daemon has initialized
    usleep(WaitTime);

    if (llvm::sys::fs::exists(SocketPath)) {
      Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
          connectToSocket(SocketPath);
      if (MaybeClient) {
        Diag.Report(diag::remark_mbd_connection) << SocketPath;
        return std::move(*MaybeClient);
      }
      consumeError(MaybeClient.takeError());
    }

    CumulativeTime += WaitTime;
    WaitTime = WaitTime * 2;
  }

  // After waiting 30 seconds give up
  return llvm::make_error<StringError>(
      "Module build daemon could not be spawned", inconvertibleErrorCode());
}

void spawnModuleBuildDaemonAndHandshake(const CompilerInvocation &Clang,
                                        DiagnosticsEngine &Diag,
                                        const char *Argv0) {

  // The module build daemon stores all output files and its socket address
  // under BasePath. Either set BasePath to a user provided option or create an
  // appropriate BasePath based on the BLAKE3 hash of the full clang version
  std::string BasePath;
  if (!Clang.getFrontendOpts().ModuleBuildDaemonPath.empty())
    BasePath = Clang.getFrontendOpts().ModuleBuildDaemonPath;
  else
    BasePath = getBasePath();

  // TODO: Max length may vary across different platforms. Incoming llvm/Support
  // for sockets will help make this portable. On most unix platforms a socket
  // address cannot be over 108 characters. The socket file, mbd.sock, takes up
  // 8 characters leaving 100 characters left for the user/system
  int MAX_ADDR = 108;
  if (BasePath.length() >= MAX_ADDR - std::string(SOCKET_FILE_NAME).length()) {
    Diag.Report(diag::err_unix_socket_addr_length)
        << BasePath << BasePath.length() << 100;
    return;
  }

  // If module build daemon does not exist spawn module build daemon
  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      getModuleBuildDaemon(Argv0, BasePath, Diag);
  if (!MaybeClient) {
    Diag.Report(diag::err_mbd_connect) << MaybeClient.takeError();
    return;
  }
  raw_socket_stream &Client = **MaybeClient;
  if (llvm::Error HandshakeErr = attemptHandshake(Client, Diag)) {
    Diag.Report(diag::err_mbd_handshake) << std::move(HandshakeErr);
    return;
  }

  Diag.Report(diag::remark_mbd_handshake);
  return;
}

} // namespace clang::tooling::cc1modbuildd
