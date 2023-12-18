//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;
using namespace clang::tooling::cc1modbuildd;

#include <errno.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdbool.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

// Create unbuffered STDOUT stream so that any logging done by the module build
// daemon can be viewed without having to terminate the process
static raw_fd_ostream &unbuff_outs() {
  static raw_fd_ostream S(fileno(stdout), false, true);
  return S;
}

static bool VerboseLog = false;
static void printVerboseLog(const llvm::Twine &message) {
  if (VerboseLog) {
    unbuff_outs() << message << '\n';
  }
}

namespace {

class ModuleBuildDaemonServer {
public:
  SmallString<256> SocketPath;
  SmallString<256> STDERR;
  SmallString<256> STDOUT;

  ModuleBuildDaemonServer(StringRef Path)
      : SocketPath(Path), STDERR(Path), STDOUT(Path) {
    llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);
    llvm::sys::path::append(STDOUT, STDOUT_FILE_NAME);
    llvm::sys::path::append(STDERR, STDERR_FILE_NAME);
  }

  ~ModuleBuildDaemonServer() { shutdownDaemon(); }

  int setupDaemonEnv();
  int createDaemonSocket();
  int listenForClients();

  static void handleClient(std::shared_ptr<raw_socket_stream> Connection);

  // TODO: modify so when shutdownDaemon is called the daemon stops accepting
  // new client connections and waits for all existing client connections to
  // terminate before closing the file descriptor and exiting
  void shutdownDaemon() {
    ServerListener.value().~ListeningSocket();
    exit(EXIT_SUCCESS);
  }

private:
  std::optional<llvm::ListeningSocket> ServerListener;
};

// Required to handle SIGTERM by calling Shutdown
ModuleBuildDaemonServer *DaemonPtr = nullptr;
void handleSignal(int Signal) {
  if (DaemonPtr != nullptr) {
    DaemonPtr->shutdownDaemon();
  }
}
} // namespace

// Sets up file descriptors and signals for module build daemon
int ModuleBuildDaemonServer::setupDaemonEnv() {

#ifdef _WIN32
  freopen("NUL", "r", stdin);
#else
  close(STDIN_FILENO);
#endif

  freopen(STDOUT.c_str(), "a", stdout);
  freopen(STDERR.c_str(), "a", stderr);

  if (signal(SIGTERM, handleSignal) == SIG_ERR) {
    errs() << "failed to handle SIGTERM" << '\n';
    exit(EXIT_FAILURE);
  }
  if (signal(SIGHUP, SIG_IGN) == SIG_ERR) {
    errs() << "failed to ignore SIGHUP" << '\n';
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}

// Creates unix socket for IPC with module build daemon
int ModuleBuildDaemonServer::createDaemonSocket() {

  Expected<ListeningSocket> MaybeServerListener =
      llvm::ListeningSocket::createUnix(SocketPath);

  if (llvm::Error Err = MaybeServerListener.takeError()) {

    llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
      // If the socket address is already in use, exit because another module
      // build daemon has successfully launched. When translation units are
      // compiled in parallel, until the socket file is created, all clang
      // invocations will try to spawn a module build daemon.
      if (std::error_code EC = SE.convertToErrorCode();
          EC == std::errc::address_in_use) {
        exit(EXIT_SUCCESS);
      } else {
        llvm::errs() << "ERROR: " << EC.message() << '\n';
        exit(EXIT_FAILURE);
      }
    });
  }

  printVerboseLog("mbd created and binded to socket at: " + SocketPath);
  ServerListener.emplace(std::move(*MaybeServerListener));
  return 0;
}

#include <cstddef>

// Function submitted to thread pool with each frontend connection. Not
// responsible for closing frontend socket connections
void ModuleBuildDaemonServer::handleClient(
    std::shared_ptr<llvm::raw_socket_stream> MovableConnection) {

  llvm::raw_socket_stream &Connection = *MovableConnection;
  std::string Buffer;

  // Read handshake from client
  if (llvm::Error ReadErr = readFromSocket(Connection, Buffer)) {
    writeError(std::move(ReadErr), "Daemon failed to read buffer from socket");
    return;
  }

  // Wait for response from module build daemon
  Expected<HandshakeMsg> MaybeHandshakeMsg =
      getSocketMsgFromBuffer<HandshakeMsg>(Buffer);
  if (!MaybeHandshakeMsg) {
    writeError(MaybeHandshakeMsg.takeError(),
               "Failed to convert buffer to HandshakeMsg: ");
    return;
  }

  // Have received HandshakeMsg - send HandshakeMsg response to clang invocation
  HandshakeMsg Msg(ActionType::HANDSHAKE, StatusType::SUCCESS);
  if (llvm::Error WriteErr = writeSocketMsgToSocket(Connection, Msg)) {
    writeError(std::move(WriteErr),
               "Failed to notify client that handshake was received");
    return;
  }

  return;
}

int ModuleBuildDaemonServer::listenForClients() {

  llvm::ThreadPool Pool;
  while (true) {

    Expected<std::unique_ptr<raw_socket_stream>> MaybeConnection =
        ServerListener.value().accept();

    if (llvm::Error Err = MaybeConnection.takeError()) {
      llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
        llvm::errs() << "ERROR: " << SE.getMessage() << '\n';
      });
      continue;
    }
    std::shared_ptr<raw_socket_stream> Connection(std::move(*MaybeConnection));
    Pool.async(handleClient, Connection);
  }
  return 0;
}

// Module build daemon is spawned with the following command line:
//
// clang -cc1modbuildd [<path>] [-v]
//
// OPTIONS
//   <path>
//       Specifies the path to all the files created by the module build daemon.
//       If provided, <path> should immediately follow -cc1modbuildd.
//
//   -v
//       Provides verbose debug information.
//
// NOTES
//     The arguments <path> and -v are optional. By default <path> follows the
//     format: /tmp/clang-<BLAKE3HashOfClangFullVersion>.
//
int cc1modbuildd_main(ArrayRef<const char *> Argv) {

  std::string BasePath;
  // command line argument parsing. -cc1modbuildd is sliced away when passing
  // Argv to cc1modbuildd_main
  if (!Argv.empty()) {

    if (find(Argv, StringRef("-v")) != Argv.end())
      VerboseLog = true;

    if (strcmp(Argv[0], "-v") != 0)
      BasePath = Argv[0];
    else
      BasePath = getBasePath();

  } else {
    BasePath = getBasePath();
  }

  // TODO: Max length may vary across different platforms. Incoming llvm/Support
  // for sockets will help make this portable. On most unix platforms a socket
  // address cannot be over 108 characters. The socket file, mbd.sock, takes up
  // 8 characters leaving 100 characters left for the user/system
  const int MAX_ADDR = 108;
  if (BasePath.length() >= MAX_ADDR - std::string(SOCKET_FILE_NAME).length()) {
    errs() << "Socket path '" + BasePath +
                  "' is too long. Socket path much be equal to or less then "
                  "100 characters. Module build daemon will not be spawned.";
    return 1;
  }

  llvm::sys::fs::create_directories(BasePath);
  ModuleBuildDaemonServer Daemon(BasePath);

  // Used to handle signals
  DaemonPtr = &Daemon;

  Daemon.setupDaemonEnv();
  Daemon.createDaemonSocket();
  Daemon.listenForClients();

  return EXIT_SUCCESS;
}
