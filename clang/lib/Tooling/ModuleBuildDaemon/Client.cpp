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

// TODO: Make portable
#if LLVM_ON_UNIX

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

using namespace clang;
using namespace llvm;
using namespace cc1modbuildd;

std::string cc1modbuildd::getBasePath() {
  llvm::BLAKE3 Hash;
  Hash.update(getClangFullVersion());
  auto HashResult = Hash.final<sizeof(uint64_t)>();
  uint64_t HashValue =
      llvm::support::endian::read<uint64_t, llvm::support::native>(
          HashResult.data());
  std::string Key = toString(llvm::APInt(64, HashValue), 36, /*Signed*/ false);

  // Set paths
  SmallString<128> BasePath;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot*/ true, BasePath);
  llvm::sys::path::append(BasePath, "clang-" + Key);
  return BasePath.c_str();
}

llvm::Error cc1modbuildd::attemptHandshake(int SocketFD,
                                           DiagnosticsEngine &Diag) {

  HandshakeMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  std::string Buffer = getBufferFromSocketMsg(Request);

  // Send HandshakeMsg to module build daemon
  Diag.Report(diag::remark_module_build_daemon)
      << "Trying to send HandshakeMsg to module build daemon";
  if (llvm::Error Err = writeToSocket(Buffer, SocketFD))
    return std::move(Err);
  Diag.Report(diag::remark_module_build_daemon)
      << "Successfully sent HandshakeMsg to module build daemon";

  // Receive response from module build daemon
  Diag.Report(diag::remark_module_build_daemon)
      << "Waiting to receive module build daemon response";
  Expected<HandshakeMsg> MaybeServerResponse =
      readSocketMsgFromSocket<HandshakeMsg>(SocketFD);
  if (!MaybeServerResponse)
    return std::move(MaybeServerResponse.takeError());
  HandshakeMsg ServerResponse = std::move(*MaybeServerResponse);

  assert(ServerResponse.MsgAction == ActionType::HANDSHAKE &&
         "Response ActionType should only ever be HANDSHAKE");

  if (ServerResponse.MsgStatus == StatusType::SUCCESS) {
    Diag.Report(diag::remark_module_build_daemon)
        << "Successfully received HandshakeMsg from module build daemon";
    return llvm::Error::success();
  }

  return llvm::make_error<StringError>(
      "Received failed handshake response from module build daemon",
      inconvertibleErrorCode());
}

llvm::Error cc1modbuildd::spawnModuleBuildDaemon(StringRef BasePath,
                                                 const char *Argv0,
                                                 DiagnosticsEngine &Diag) {
  std::string BasePathStr = BasePath.str();
  const char *Args[] = {Argv0, "-cc1modbuildd", BasePathStr.c_str(), nullptr};
  pid_t pid;
  Diag.Report(diag::remark_module_build_daemon)
      << "Trying to spawn module build daemon";
  int EC = posix_spawn(&pid, Args[0],
                       /*file_actions*/ nullptr,
                       /*spawnattr*/ nullptr, const_cast<char **>(Args),
                       /*envp*/ nullptr);
  if (EC)
    return createStringError(std::error_code(EC, std::generic_category()),
                             "Failed to spawn module build daemon");

  Diag.Report(diag::remark_module_build_daemon)
      << "Successfully spawned module build daemon";
  return llvm::Error::success();
}

Expected<int> cc1modbuildd::getModuleBuildDaemon(const char *Argv0,
                                                 StringRef BasePath,
                                                 DiagnosticsEngine &Diag) {

  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  if (llvm::sys::fs::exists(SocketPath)) {
    Expected<int> MaybeFD = connectToSocket(SocketPath);
    if (MaybeFD) {
      Diag.Report(diag::remark_module_build_daemon)
          << "Module build daemon already exists";
      return std::move(*MaybeFD);
    }
    consumeError(MaybeFD.takeError());
  }

  if (llvm::Error Err =
          cc1modbuildd::spawnModuleBuildDaemon(BasePath, Argv0, Diag))
    return std::move(Err);

  const unsigned int MICROSEC_IN_SEC = 1000000;
  constexpr unsigned int MAX_WAIT_TIME = 30 * MICROSEC_IN_SEC;

  unsigned int CumulativeTime = 0;
  unsigned int WaitTime = 10;

  while (CumulativeTime <= MAX_WAIT_TIME) {
    // Wait a bit then check to see if the module build daemon has initialized
    usleep(WaitTime);

    Diag.Report(diag::remark_module_build_daemon)
        << "Trying to connect to recently spawned module build daemon";
    if (llvm::sys::fs::exists(SocketPath)) {
      Expected<int> MaybeFD = connectToSocket(SocketPath);
      if (MaybeFD) {
        Diag.Report(diag::remark_module_build_daemon)
            << "Succesfully connected to recently spawned module build daemon";
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

void cc1modbuildd::spawnModuleBuildDaemonAndHandshake(
    const CompilerInvocation &Clang, DiagnosticsEngine &Diag,
    const char *Argv0) {

  // The module build daemon stores all output files and its socket address
  // under BasePath. Either set BasePath to a user provided option or create an
  // appropriate BasePath based on the hash of the full clang version
  std::string BasePath;
  if (!Clang.getFrontendOpts().ModuleBuildDaemonPath.empty())
    BasePath = Clang.getFrontendOpts().ModuleBuildDaemonPath;
  else
    BasePath = cc1modbuildd::getBasePath();

  // On most unix platforms a socket address cannot be over 108 characters
  int MAX_ADDR = 108;
  if (BasePath.length() >= MAX_ADDR - std::string(SOCKET_FILE_NAME).length()) {
    Diag.Report(diag::warn_module_build_daemon)
        << "Provided socket path" + BasePath +
               " is too long. Socket path much be equal to or less then 100 "
               "characters. Invocation will not spawn module build daemon.";
    return;
  }

  // If module build daemon does not exist spawn module build daemon
  Expected<int> MaybeDaemonFD =
      cc1modbuildd::getModuleBuildDaemon(Argv0, BasePath, Diag);
  if (!MaybeDaemonFD) {
    Diag.Report(diag::warn_module_build_daemon) << getFullErrorMsg(
        MaybeDaemonFD.takeError(), "Attempt to connect to daemon has failed: ");
    return;
  }
  int DaemonFD = std::move(*MaybeDaemonFD);

  if (llvm::Error HandshakeErr = attemptHandshake(DaemonFD, Diag)) {
    Diag.Report(diag::warn_module_build_daemon) << getFullErrorMsg(
        std::move(HandshakeErr), "Attempted hadshake with daemon has failed: ");
    return;
  }

  Diag.Report(diag::remark_module_build_daemon)
      << "Completed successfull handshake with module build daemon";
  return;
}

#endif // LLVM_ON_UNIX
