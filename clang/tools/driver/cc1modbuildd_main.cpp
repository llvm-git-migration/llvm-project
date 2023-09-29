//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <errno.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <signal.h>
#include <sstream>
#include <stdbool.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>

using namespace llvm;
using namespace clang;
using namespace cc1modbuildd;

// Create unbuffered STDOUT stream so that any logging done by module build
// daemon can be viewed without having to terminate the process
static raw_fd_ostream &unbuff_outs() {
  static raw_fd_ostream S(STDOUT_FILENO, false, true);
  return S;
}

namespace {

struct ClientConnection {
  int ClientFD;
  std::string Buffer;
};

class ModuleBuildDaemonServer {
public:
  SmallString<256> SocketPath;
  SmallString<256> STDERR;
  SmallString<256> STDOUT;

  ModuleBuildDaemonServer(StringRef Path, ArrayRef<const char *> Argv)
      : SocketPath(Path), STDERR(Path), STDOUT(Path) {
    llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);
    llvm::sys::path::append(STDOUT, STDOUT_FILE_NAME);
    llvm::sys::path::append(STDERR, STDERR_FILE_NAME);
  }

  ~ModuleBuildDaemonServer() { shutdownDaemon(); }

  int forkDaemon();
  int createDaemonSocket();
  int listenForClients();

  static void handleClient(ClientConnection Connection);

  void shutdownDaemon() {
    int SocketFD = ListenSocketFD.load();

    unlink(SocketPath.c_str());
    shutdown(SocketFD, SHUT_RD);
    close(SocketFD);
    exit(EXIT_SUCCESS);
  }

private:
  pid_t Pid = -1;
  std::atomic<int> ListenSocketFD = -1;
};

// Required to handle SIGTERM by calling Shutdown
ModuleBuildDaemonServer *DaemonPtr = nullptr;
void handleSignal(int Signal) {
  if (DaemonPtr != nullptr) {
    DaemonPtr->shutdownDaemon();
  }
}
} // namespace

static bool VerboseLog = false;
static void printVerboseLog(const llvm::Twine &message) {
  if (VerboseLog) {
    unbuff_outs() << message << '\n';
  }
}

// Forks and detaches process, creating module build daemon
int ModuleBuildDaemonServer::forkDaemon() {

  pid_t pid = fork();

  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  Pid = getpid();

  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);

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
  if (setsid() == -1) {
    errs() << "setsid failed" << '\n';
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

// Creates unix socket for IPC with module build daemon
int ModuleBuildDaemonServer::createDaemonSocket() {

  // New socket
  int SocketFD = socket(AF_UNIX, SOCK_STREAM, 0);

  if (SocketFD == -1) {
    std::perror("Socket create error: ");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SocketPath.c_str(), sizeof(addr.sun_path) - 1);

  // bind to local address
  if (bind(SocketFD, (struct sockaddr *)&addr, sizeof(addr)) == -1) {

    // If the socket address is already in use, exit because another module
    // build daemon has successfully launched. When translation units are
    // compiled in parallel, until the socket file is created, all clang
    // invocations will spawn a module build daemon.
    if (errno == EADDRINUSE) {
      close(SocketFD);
      exit(EXIT_SUCCESS);
    }
    std::perror("Socket bind error: ");
    exit(EXIT_FAILURE);
  }
  printVerboseLog("mbd created and binded to socket address at: " + SocketPath);

  // set socket to accept incoming connection request
  unsigned MaxBacklog = llvm::hardware_concurrency().compute_thread_count();
  if (listen(SocketFD, MaxBacklog) == -1) {
    std::perror("Socket listen error: ");
    exit(EXIT_FAILURE);
  }

  ListenSocketFD.store(SocketFD);
  return 0;
}

// Function submitted to thread pool with each client connection. Not
// responsible for closing client connections
// TODO: Setup something like ScopedHandle to auto close client on return
void ModuleBuildDaemonServer::handleClient(ClientConnection Connection) {

  // Read handshake from client
  if (llvm::Error ReadErr =
          readFromSocket(Connection.ClientFD, Connection.Buffer)) {
    writeError(std::move(ReadErr), "Daemon failed to read buffer from socket");
    close(Connection.ClientFD);
    return;
  }

  // Wait for response from module build daemon
  Expected<HandshakeMsg> MaybeHandshakeMsg =
      getSocketMsgFromBuffer<HandshakeMsg>(Connection.Buffer);
  if (!MaybeHandshakeMsg) {
    writeError(MaybeHandshakeMsg.takeError(),
               "Failed to convert buffer to HandshakeMsg: ");
    close(Connection.ClientFD);
    return;
  }

  // Have received HandshakeMsg - send HandshakeMsg response to clang invocation
  HandshakeMsg Msg(ActionType::HANDSHAKE, StatusType::SUCCESS);
  if (llvm::Error WriteErr = writeSocketMsgToSocket(Msg, Connection.ClientFD)) {
    writeError(std::move(WriteErr),
               "Failed to notify client that handshake was received");
    close(Connection.ClientFD);
    return;
  }

  // Remove HandshakeMsg from Buffer in preperation for next read. Not currently
  // necessary but will be once Daemon increases communication
  size_t Position = Connection.Buffer.find("...\n");
  if (Position != std::string::npos)
    Connection.Buffer = Connection.Buffer.substr(Position + 4);

  close(Connection.ClientFD);
  return;
}

int ModuleBuildDaemonServer::listenForClients() {

  llvm::ThreadPool Pool;
  int Client;

  while (true) {

    if ((Client = accept(ListenSocketFD.load(), NULL, NULL)) == -1) {
      std::perror("Socket accept error: ");
      continue;
    }

    ClientConnection Connection;
    Connection.ClientFD = Client;

    Pool.async(handleClient, Connection);
  }
  return 0;
}

// Module build daemon is spawned with the following command line:
//
// clang -cc1modbuildd <path> -v
//
// <path> defines the location of all files created by the module build daemon
// and should follow the format /path/to/dir. For example, `clang
// -cc1modbuildd /tmp/` creates a socket file at `/tmp/mbd.sock`
//
// When a module build daemon is spawned by a cc1 invocations, <path> follows
// the format /tmp/clang-<BLAKE3HashOfClangFullVersion> and looks something like
// /tmp/clang-3NXKISKJ0WJTN
//
// -v is optional and provides berbose debug information
//
int cc1modbuildd_main(ArrayRef<const char *> Argv) {

  if (Argv.size() < 1) {
    errs() << "spawning a module build daemon requies a command line format of "
              "`clang -cc1modbuildd <path>`. <path> defines where the module "
              "build daemon will create mbd.out, mbd.err, mbd.sock"
           << '\n';
    return 1;
  }

  // Where to store log files and socket address
  // TODO: Add check to confirm BasePath is directory
  std::string BasePath(Argv[0]);

  // On most unix platforms a socket address cannot be over 108 characters
  int MAX_ADDR = 108;
  if (BasePath.length() >= MAX_ADDR - std::string(SOCKET_FILE_NAME).length()) {
    errs() << "Provided socket path" + BasePath +
                  " is too long. Socket path much be equal to or less then 100 "
                  "characters. Module build daemon will not be spawned.";
    return 1;
  }

  llvm::sys::fs::create_directories(BasePath);
  ModuleBuildDaemonServer Daemon(BasePath, Argv);

  // Used to handle signals
  DaemonPtr = &Daemon;

  if (find(Argv, StringRef("-v")) != Argv.end())
    VerboseLog = true;

  Daemon.forkDaemon();
  Daemon.createDaemonSocket();
  Daemon.listenForClients();

  return 0;
}

#endif // LLVM_ON_UNIX
