//===-- lldb-platform.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cerrno>
#if defined(__APPLE__)
#include <netinet/in.h>
#endif
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#include <fstream>
#include <optional>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "Acceptor.h"
#include "LLDBServerUtilities.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServerPlatform.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/HostGetOpt.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace lldb_private::process_gdb_remote;
using namespace llvm;

// option descriptors for getopt_long_only()

static int g_debug = 0;
static int g_verbose = 0;
static int g_server = 0;
static volatile bool g_listen_gdb = true;

static struct option g_long_options[] = {
    {"debug", no_argument, &g_debug, 1},
    {"verbose", no_argument, &g_verbose, 1},
    {"log-file", required_argument, nullptr, 'l'},
    {"log-channels", required_argument, nullptr, 'c'},
    {"listen", required_argument, nullptr, 'L'},
    {"port-offset", required_argument, nullptr, 'p'},
    {"gdbserver-port", required_argument, nullptr, 'P'},
    {"min-gdbserver-port", required_argument, nullptr, 'm'},
    {"max-gdbserver-port", required_argument, nullptr, 'M'},
    {"socket-file", required_argument, nullptr, 'f'},
    {"server", no_argument, &g_server, 1},
    {"child-platform-fd", required_argument, nullptr, 2},
    {nullptr, 0, nullptr, 0}};

#if defined(__APPLE__)
#define LOW_PORT (IPPORT_RESERVED)
#define HIGH_PORT (IPPORT_HIFIRSTAUTO)
#else
#define LOW_PORT (1024u)
#define HIGH_PORT (65535u)
#endif

#if !defined(_WIN32)
// Watch for signals
static void signal_handler(int signo) {
  switch (signo) {
  case SIGHUP:
    // Use SIGINT first, if that does not work, use SIGHUP as a last resort.
    // And we should not call exit() here because it results in the global
    // destructors to be invoked and wreaking havoc on the threads still
    // running.
    llvm::errs() << "SIGHUP received, exiting lldb-server...\n";
    abort();
    break;
  }
}
#endif

static void display_usage(const char *progname, const char *subcommand) {
  fprintf(stderr, "Usage:\n  %s %s [--log-file log-file-name] [--log-channels "
                  "log-channel-list] [--port-file port-file-path] --server "
                  "--listen port\n",
          progname, subcommand);
  exit(0);
}

static Status save_socket_id_to_file(const std::string &socket_id,
                                     const FileSpec &file_spec) {
  FileSpec temp_file_spec(file_spec.GetDirectory().GetStringRef());
  Status error(llvm::sys::fs::create_directory(temp_file_spec.GetPath()));
  if (error.Fail())
    return Status("Failed to create directory %s: %s",
                  temp_file_spec.GetPath().c_str(), error.AsCString());

  Status status;
  if (auto Err = llvm::writeToOutput(file_spec.GetPath(),
                                     [&socket_id](llvm::raw_ostream &OS) {
                                       OS << socket_id;
                                       return llvm::Error::success();
                                     }))
    return Status("Failed to atomically write file %s: %s",
                  file_spec.GetPath().c_str(),
                  llvm::toString(std::move(Err)).c_str());
  return status;
}

static void client_handle(GDBRemoteCommunicationServerPlatform &platform,
                          const lldb_private::Args &args) {
  if (!platform.IsConnected())
    return;

  if (args.GetArgumentCount() > 0) {
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
    std::string socket_name;
    Status error = platform.LaunchGDBServer(args, pid, socket_name,
                                            SharedSocket::kInvalidFD);
    if (error.Success())
      platform.SetPendingGdbServer(socket_name);
    else
      fprintf(stderr, "failed to start gdbserver: %s\n", error.AsCString());
  }

  bool interrupt = false;
  bool done = false;
  Status error;
  while (!interrupt && !done) {
    if (platform.GetPacketAndSendResponse(std::nullopt, error, interrupt,
                                          done) !=
        GDBRemoteCommunication::PacketResult::Success)
      break;
  }

  printf("Disconnected.\n");
}

static void spawn_process_reaped(lldb::pid_t pid, int signal, int status) {}

static Status spawn_process(const char *progname, Connection *conn,
                            uint16_t gdb_port, uint16_t port_offset,
                            const lldb_private::Args &args,
                            const std::string &log_file,
                            const StringRef log_channels) {
  Status error;
  SharedSocket shared_socket(conn, error);
  if (error.Fail())
    return error;

  ProcessLaunchInfo launch_info;

  FileSpec self_spec(progname, FileSpec::Style::native);
  launch_info.SetExecutableFile(self_spec, true);
  Args &self_args = launch_info.GetArguments();
  self_args.AppendArgument(llvm::StringRef("platform"));
  self_args.AppendArgument(llvm::StringRef("--child-platform-fd"));
  self_args.AppendArgument(llvm::to_string(shared_socket.GetSendableFD()));

  // Ignored in Windows
  launch_info.AppendDuplicateFileAction((int)shared_socket.GetSendableFD(),
                                        (int)shared_socket.GetSendableFD());

  if (gdb_port) {
    self_args.AppendArgument(llvm::StringRef("--gdbserver-port"));
    self_args.AppendArgument(llvm::to_string(gdb_port));
  }
  if (port_offset > 0) {
    self_args.AppendArgument(llvm::StringRef("--port-offset"));
    self_args.AppendArgument(llvm::to_string(port_offset));
  }
  if (!log_file.empty()) {
    self_args.AppendArgument(llvm::StringRef("--log-file"));
    self_args.AppendArgument(log_file);
  }
  if (!log_channels.empty()) {
    self_args.AppendArgument(llvm::StringRef("--log-channels"));
    self_args.AppendArgument(log_channels);
  }
  if (args.GetArgumentCount() > 0) {
    self_args.AppendArgument("--");
    self_args.AppendArguments(args);
  }

  launch_info.SetLaunchInSeparateProcessGroup(false);
  launch_info.SetMonitorProcessCallback(&spawn_process_reaped);

  // Copy the current environment.
  launch_info.GetEnvironment() = Host::GetEnvironment();

  launch_info.GetFlags().Set(eLaunchFlagDisableSTDIO);

  // Close STDIN, STDOUT and STDERR.
  launch_info.AppendCloseFileAction(STDIN_FILENO);
  launch_info.AppendCloseFileAction(STDOUT_FILENO);
  launch_info.AppendCloseFileAction(STDERR_FILENO);

  // Redirect STDIN, STDOUT and STDERR to "/dev/null".
  launch_info.AppendSuppressFileAction(STDIN_FILENO, true, false);
  launch_info.AppendSuppressFileAction(STDOUT_FILENO, false, true);
  launch_info.AppendSuppressFileAction(STDERR_FILENO, false, true);

  std::string cmd;
  self_args.GetCommandString(cmd);

  error = Host::LaunchProcess(launch_info);
  if (error.Fail())
    return error;

  lldb::pid_t child_pid = launch_info.GetProcessID();
  if (child_pid == LLDB_INVALID_PROCESS_ID)
    return Status("invalid pid");

  LLDB_LOG(GetLog(LLDBLog::Platform), "lldb-platform launched '{0}', pid={1}",
           cmd, child_pid);

  error = shared_socket.CompleteSending(child_pid);
  if (error.Fail()) {
    Host::Kill(child_pid, SIGTERM);
    return error;
  }

  return Status();
}

static thread_result_t
gdb_thread_proc(Acceptor *acceptor_gdb, uint16_t gdbserver_port,
                const lldb_private::Args &inferior_arguments) {
  lldb_private::Args args(inferior_arguments);

  Log *log = GetLog(LLDBLog::Platform);
  GDBRemoteCommunicationServerPlatform platform(Socket::ProtocolTcp,
                                                gdbserver_port);
  while (g_listen_gdb) {
    Connection *conn = nullptr;
    Status error =
        acceptor_gdb->Accept(/*children_inherit_accept_socket=*/false, conn);
    if (error.Fail()) {
      WithColor::error() << error.AsCString() << '\n';
      break;
    }
    printf("gdb connection established.\n");

    {
      SharedSocket shared_socket(conn, error);
      if (error.Success()) {
        lldb::pid_t child_pid = LLDB_INVALID_PROCESS_ID;
        std::string socket_name;
        error = platform.LaunchGDBServer(args, child_pid, socket_name,
                                         shared_socket.GetSendableFD());
        if (error.Fail()) {
          LLDB_LOGF(log, "gdbserver LaunchGDBServer failed: %s",
                    error.AsCString());
        } else if (child_pid != LLDB_INVALID_PROCESS_ID) {
          error = shared_socket.CompleteSending(child_pid);
          if (error.Success()) {
            // Use inferior arguments once.
            args.Clear();
          } else {
            Host::Kill(child_pid, SIGTERM);
            LLDB_LOGF(log, "gdbserver CompleteSending failed: %s",
                      error.AsCString());
          }
        }
      } else
        LLDB_LOGF(log, "gdbserver SharedSocket failed: %s", error.AsCString());
    }
    delete conn;
  }
  return {};
}

// main
int main_platform(int argc, char *argv[]) {
  const char *progname = argv[0];
  const char *subcommand = argv[1];
  argc--;
  argv++;
#if !defined(_WIN32)
  signal(SIGPIPE, SIG_IGN);
  signal(SIGHUP, signal_handler);
#endif
  int long_option_index = 0;
  Status error;
  std::string listen_host_port;
  int ch;

  std::string log_file;
  StringRef
      log_channels; // e.g. "lldb process threads:gdb-remote default:linux all"

  shared_fd_t fd = SharedSocket::kInvalidFD;

  uint16_t gdbserver_port = 0;
  uint16_t port_offset = 0;

  FileSpec socket_file;
  bool show_usage = false;
  int option_error = 0;
  int socket_error = -1;

  std::string short_options(OptionParser::GetShortOptionString(g_long_options));

#if __GLIBC__
  optind = 0;
#else
  optreset = 1;
  optind = 1;
#endif

  while ((ch = getopt_long_only(argc, argv, short_options.c_str(),
                                g_long_options, &long_option_index)) != -1) {
    switch (ch) {
    case 0: // Any optional that auto set themselves will return 0
      break;

    case 'L':
      listen_host_port.append(optarg);
      break;

    case 'l': // Set Log File
      if (optarg && optarg[0])
        log_file.assign(optarg);
      break;

    case 'c': // Log Channels
      if (optarg && optarg[0])
        log_channels = StringRef(optarg);
      break;

    case 'f': // Socket file
      if (optarg && optarg[0])
        socket_file.SetFile(optarg, FileSpec::Style::native);
      break;

    case 'p': {
      if (!llvm::to_integer(optarg, port_offset)) {
        WithColor::error() << "invalid port offset string " << optarg << "\n";
        option_error = 4;
        break;
      }
      if (port_offset < LOW_PORT || port_offset > HIGH_PORT) {
        WithColor::error() << llvm::formatv(
            "port offset {0} is not in the "
            "valid user port range of {1} - {2}\n",
            port_offset, LOW_PORT, HIGH_PORT);
        option_error = 5;
      }
    } break;

    case 'P':
    case 'm':
    case 'M': {
      uint16_t portnum;
      if (!llvm::to_integer(optarg, portnum)) {
        WithColor::error() << "invalid port number string " << optarg << "\n";
        option_error = 2;
        break;
      }
      if (portnum < LOW_PORT || portnum > HIGH_PORT) {
        WithColor::error() << llvm::formatv(
            "port number {0} is not in the "
            "valid user port range of {1} - {2}\n",
            portnum, LOW_PORT, HIGH_PORT);
        option_error = 1;
        break;
      }
      if (ch == 'P')
        gdbserver_port = portnum;
      else if (gdbserver_port == 0)
        gdbserver_port = portnum;
    } break;

    case 2: {
      uint64_t _fd;
      if (!llvm::to_integer(optarg, _fd)) {
        WithColor::error() << "invalid fd " << optarg << "\n";
        option_error = 6;
      } else
        fd = (shared_fd_t)_fd;
    } break;

    case 'h': /* fall-through is intentional */
    case '?':
      show_usage = true;
      break;
    }
  }

  if (!LLDBServerUtilities::SetupLogging(log_file, log_channels, 0))
    return -1;

  // Print usage and exit if no listening port is specified.
  if (listen_host_port.empty() && fd == SharedSocket::kInvalidFD)
    show_usage = true;

  if (show_usage || option_error) {
    display_usage(progname, subcommand);
    exit(option_error);
  }

  // Skip any options we consumed with getopt_long_only.
  argc -= optind;
  argv += optind;
  lldb_private::Args inferior_arguments;
  inferior_arguments.SetArguments(argc, const_cast<const char **>(argv));

  if (fd != SharedSocket::kInvalidFD) {
    // Child process will handle the connection and exit.
    Log *log = GetLog(LLDBLog::Platform);
    if (!listen_host_port.empty()) {
      LLDB_LOGF(log, "lldb-platform child: "
                     "ambiguous parameters --listen and --child-platform-fd");
      return socket_error;
    }

    if (gdbserver_port == 0) {
      LLDB_LOGF(log, "lldb-platform child: "
                     "--gdbserver-port is missing.");
      return socket_error;
    }

    NativeSocket socket;
    error = SharedSocket::GetNativeSocket(fd, socket);
    if (error.Fail()) {
      LLDB_LOGF(log, "lldb-platform child: %s", error.AsCString());
      return socket_error;
    }

    Connection *conn =
        new ConnectionFileDescriptor(new TCPSocket(socket, true, false));
    GDBRemoteCommunicationServerPlatform platform(Socket::ProtocolTcp,
                                                  gdbserver_port);
    if (port_offset > 0)
      platform.SetPortOffset(port_offset);
    platform.SetConnection(std::unique_ptr<Connection>(conn));
    client_handle(platform, inferior_arguments);
    return 0;
  }

  const bool children_inherit_listen_socket = false;
  // the test suite makes many connections in parallel, let's not miss any.
  // The highest this should get reasonably is a function of the number
  // of target CPUs. For now, let's just use 100.
  const int backlog = 100;

  std::unique_ptr<Acceptor> acceptor_up(Acceptor::Create(
      listen_host_port, children_inherit_listen_socket, error));
  if (error.Fail()) {
    fprintf(stderr, "failed to create acceptor: %s", error.AsCString());
    exit(socket_error);
  }

  if (g_server && acceptor_up->GetSocketProtocol() != Socket::ProtocolTcp) {
    fprintf(stderr,
            "ambiguous parameters --server --listen %s\n"
            "The protocol must be tcp for the server mode.",
            listen_host_port.c_str());
    exit(socket_error);
  }

  std::unique_ptr<Acceptor> acceptor_gdb;
  HostThread gdb_thread;
  if (acceptor_up->GetSocketProtocol() == Socket::ProtocolTcp) {
    // Use the same host from listen_host_port for acceptor_gdb.
    std::size_t port_pos = listen_host_port.rfind(":");
    if (port_pos == std::string::npos)
      port_pos = 0;
    else
      ++port_pos;
    std::string listen_gdb_port = llvm::formatv(
        "{0}{1}", listen_host_port.substr(0, port_pos), gdbserver_port);
    acceptor_gdb = Acceptor::Create(listen_gdb_port,
                                    children_inherit_listen_socket, error);
    if (error.Fail()) {
      fprintf(stderr, "failed to create gdb acceptor: %s", error.AsCString());
      exit(socket_error);
    }

    error = acceptor_gdb->Listen(backlog);
    if (error.Fail()) {
      printf("failed to listen gdb: %s\n", error.AsCString());
      exit(socket_error);
    }

    if (gdbserver_port == 0) {
      std::string str_gdbserver_port = acceptor_gdb->GetLocalSocketId();
      if (!llvm::to_integer(str_gdbserver_port, gdbserver_port)) {
        printf("invalid gdb port: %s\n", str_gdbserver_port.c_str());
        exit(socket_error);
      }
      Log *log = GetLog(LLDBLog::Connection);
      LLDB_LOG(log, "Listen to gdb port {0}", gdbserver_port);
    }
    assert(gdbserver_port);

    const std::string thread_name = llvm::formatv("gdb:{0}", gdbserver_port);
    Acceptor *ptr_acceptor_gdb = acceptor_gdb.get();
    auto maybe_thread = ThreadLauncher::LaunchThread(
        thread_name, [ptr_acceptor_gdb, gdbserver_port, inferior_arguments] {
          return gdb_thread_proc(ptr_acceptor_gdb, gdbserver_port,
                                 inferior_arguments);
        });
    if (!maybe_thread) {
      WithColor::error() << "failed to start gdb listen thread: "
                         << maybe_thread.takeError() << '\n';
      exit(socket_error);
    }
    gdb_thread = *maybe_thread;
  }

  error = acceptor_up->Listen(backlog);
  if (error.Fail()) {
    printf("failed to listen: %s\n", error.AsCString());
    exit(socket_error);
  }
  if (socket_file) {
    error =
        save_socket_id_to_file(acceptor_up->GetLocalSocketId(), socket_file);
    if (error.Fail()) {
      fprintf(stderr, "failed to write socket id to %s: %s\n",
              socket_file.GetPath().c_str(), error.AsCString());
      return 1;
    }
  }

  GDBRemoteCommunicationServerPlatform platform(
      acceptor_up->GetSocketProtocol(), gdbserver_port);
  if (port_offset > 0)
    platform.SetPortOffset(port_offset);

  do {
    const bool children_inherit_accept_socket = true;
    Connection *conn = nullptr;
    error = acceptor_up->Accept(children_inherit_accept_socket, conn);
    if (error.Fail()) {
      WithColor::error() << error.AsCString() << '\n';
      exit(socket_error);
    }
    printf("Connection established.\n");

    if (g_server) {
      error = spawn_process(progname, conn, gdbserver_port, port_offset,
                            inferior_arguments, log_file, log_channels);
      if (error.Fail()) {
        LLDB_LOGF(GetLog(LLDBLog::Platform), "spawn_process failed: %s",
                  error.AsCString());
        WithColor::error() << "spawn_process failed: " << error.AsCString()
                           << "\n";
      }
      // Parent doesn't need a connection to the lldb client
      delete conn;

      // Parent will continue to listen for new connections.
      continue;
    } else {
      // If not running as a server, this process will not accept
      // connections while a connection is active.
      acceptor_up.reset();
    }

    platform.SetConnection(std::unique_ptr<Connection>(conn));
    client_handle(platform, inferior_arguments);
  } while (g_server);

  // FIXME: Make TCPSocket::CloseListenSockets() public and implement
  // Acceptor::Close().
  /*
  if (acceptor_gdb && gdb_thread.IsJoinable()) {
    g_listen_gdb = false;
    static_cast<TCPSocket *>(acceptor_gdb->m_listener_socket_up.get())
        ->CloseListenSockets();
    lldb::thread_result_t result;
    gdb_thread.Join(&result);
  }
  */

  fprintf(stderr, "lldb-server exiting...\n");

  return 0;
}
