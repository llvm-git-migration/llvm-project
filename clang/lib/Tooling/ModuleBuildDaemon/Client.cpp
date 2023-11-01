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
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Program.h"

// TODO: Make portable
#if LLVM_ON_UNIX

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

llvm::Error attemptHandshake(int SocketFD, DiagnosticsEngine &Diag) {

  HandshakeMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  std::string Buffer = getBufferFromSocketMsg(Request);

  // Send HandshakeMsg to module build daemon
  if (llvm::Error Err = writeToSocket(Buffer, SocketFD))
    return std::move(Err);

  // Receive response from module build daemon
  Expected<HandshakeMsg> MaybeServerResponse =
      readSocketMsgFromSocket<HandshakeMsg>(SocketFD);
  if (!MaybeServerResponse)
    return std::move(MaybeServerResponse.takeError());
  HandshakeMsg ServerResponse = std::move(*MaybeServerResponse);

  assert(ServerResponse.MsgAction == ActionType::HANDSHAKE &&
         "Response ActionType should only ever be HANDSHAKE");

  if (ServerResponse.MsgStatus == StatusType::SUCCESS) {
    return llvm::Error::success();
  }

  return llvm::make_error<StringError>(
      "Received failed handshake response from module build daemon",
      inconvertibleErrorCode());
}

llvm::Error spawnModuleBuildDaemon(StringRef BasePath, const char *Argv0,
                                   DiagnosticsEngine &Diag) {

  std::vector<StringRef> Args = {Argv0, "-cc1modbuildd", BasePath.str()};

  std::string ErrorBuffer;
  // Will wait until module build daemon has forked and parent process. There
  // is extra work that needs to be done for Windows when using ExecuteNoWait
  llvm::sys::ExecuteAndWait(Argv0, Args, std::nullopt, {}, 0, 0, &ErrorBuffer);

  if (!ErrorBuffer.empty())
    return llvm::make_error<StringError>(ErrorBuffer, inconvertibleErrorCode());

  Diag.Report(diag::remark_mbd_spawn);
  return llvm::Error::success();
}

Expected<int> getModuleBuildDaemon(const char *Argv0, StringRef BasePath,
                                   DiagnosticsEngine &Diag) {

  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<int> MaybeFD = connectToSocket(SocketPath);
    if (MaybeFD) {
      return std::move(*MaybeFD);
    }
    consumeError(MaybeFD.takeError());
  }

  if (llvm::Error Err = spawnModuleBuildDaemon(BasePath, Argv0, Diag))
    return std::move(Err);

  const unsigned int MICROSEC_IN_SEC = 1000000;
  constexpr unsigned int MAX_WAIT_TIME = 30 * MICROSEC_IN_SEC;

  unsigned int CumulativeTime = 0;
  unsigned int WaitTime = 10;

  while (CumulativeTime <= MAX_WAIT_TIME) {
    // Wait a bit then check to see if the module build daemon has initialized
    usleep(WaitTime);

    if (llvm::sys::fs::exists(SocketPath)) {
      Expected<int> MaybeFD = connectToSocket(SocketPath);
      if (MaybeFD) {
        Diag.Report(diag::remark_mbd_connection) << SocketPath;
        return std::move(*MaybeFD);
      }
      consumeError(MaybeFD.takeError());
    }

    CumulativeTime += WaitTime;
    // Exponential backoff
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
  Expected<int> MaybeDaemonFD = getModuleBuildDaemon(Argv0, BasePath, Diag);
  if (!MaybeDaemonFD) {
    Diag.Report(diag::err_mbd_connect) << MaybeDaemonFD.takeError();
    return;
  }
  int DaemonFD = std::move(*MaybeDaemonFD);

  if (llvm::Error HandshakeErr = attemptHandshake(DaemonFD, Diag)) {
    Diag.Report(diag::err_mbd_handshake) << std::move(HandshakeErr);
    return;
  }

  Diag.Report(diag::remark_mbd_handshake);
  return;
}

} // namespace clang::tooling::cc1modbuildd

#endif // LLVM_ON_UNIX
