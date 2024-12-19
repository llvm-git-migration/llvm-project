//===-- Statusline.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Statusline.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include <sys/ioctl.h>
#include <termios.h>

#define ESCAPE "\x1b"
#define ANSI_SAVE_CURSOR ESCAPE "7"
#define ANSI_RESTORE_CURSOR ESCAPE "8"
#define ANSI_CLEAR_BELOW ESCAPE "[J"
#define ANSI_CLEAR_LINE "\r\x1B[2K"
#define ANSI_SET_SCROLL_ROWS ESCAPE "[0;%ur"
#define ANSI_TO_START_OF_ROW ESCAPE "[%u;0f"
#define ANSI_UP_ROWS ESCAPE "[%dA"
#define ANSI_DOWN_ROWS ESCAPE "[%dB"
#define ANSI_FORWARD_COLS ESCAPE "\033[%dC"
#define ANSI_BACKWARD_COLS ESCAPE "\033[%dD"

using namespace lldb;
using namespace lldb_private;

Statusline::Statusline(Debugger &debugger) : m_debugger(debugger) {}

Statusline::~Statusline() { StopStatuslineThread(); }

bool Statusline::IsSupported() const {
  File &file = m_debugger.GetOutputFile();
  return file.GetIsInteractive() && file.GetIsTerminalWithColors();
}

void Statusline::Enable() {
  if (!IsSupported())
    return;

  UpdateTerminalProperties();
  // Reduce the scroll window to make space for the status bar below.
  SetScrollWindow(m_terminal_height - 1);
}

void Statusline::Disable() {
  UpdateTerminalProperties();
  // Clear the previous status bar if any.
  Clear();
  // Extend the scroll window to cover the status bar.
  SetScrollWindow(m_terminal_height);
}

void Statusline::Draw(llvm::StringRef str) {
  UpdateTerminalProperties();

  const uint32_t ellipsis = 3;
  if (str.size() + ellipsis >= m_terminal_width)
    str = str.substr(0, m_terminal_width - ellipsis);

  StreamFile &out = m_debugger.GetOutputStream();
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_TO_START_OF_ROW, static_cast<unsigned>(m_terminal_height));
  out << ANSI_CLEAR_LINE;
  out << ansi::FormatAnsiTerminalCodes(m_ansi_prefix);
  out << str;
  out << std::string(m_terminal_width - str.size(), ' ');
  out << ansi::FormatAnsiTerminalCodes(m_ansi_suffix);
  out << ANSI_RESTORE_CURSOR;
}

void Statusline::Reset() {
  StreamFile &out = m_debugger.GetOutputStream();
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_TO_START_OF_ROW, static_cast<unsigned>(m_terminal_height));
  out << ANSI_CLEAR_LINE;
  out << ANSI_RESTORE_CURSOR;
}

void Statusline::Clear() { Draw(""); }

void Statusline::UpdateTerminalProperties() {
  if (m_terminal_size_has_changed == 0)
    return;

  // Clear the previous statusline.
  Reset();

  // Purposely ignore the terminal settings. If the setting doesn't match
  // reality and we draw the status bar over existing text, we have no way to
  // recover.
  struct winsize window_size;
  if ((isatty(STDIN_FILENO) != 0) &&
      ::ioctl(STDIN_FILENO, TIOCGWINSZ, &window_size) == 0) {
    m_terminal_width = window_size.ws_col;
    m_terminal_height = window_size.ws_row;
  }

  // Set the scroll window based on the new terminal height.
  SetScrollWindow(m_terminal_height - 1);

  // Clear the flag.
  m_terminal_size_has_changed = 0;
}

void Statusline::SetScrollWindow(uint64_t height) {
  StreamFile &out = m_debugger.GetOutputStream();
  out << '\n';
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_SET_SCROLL_ROWS, static_cast<unsigned>(height));
  out << ANSI_RESTORE_CURSOR;
  out.Printf(ANSI_UP_ROWS, 1);
  out << ANSI_CLEAR_BELOW;
  out.Flush();

  m_scroll_height = height;
}

lldb::thread_result_t Statusline::StatuslineThread() {
  using namespace std::chrono_literals;
  static constexpr const std::chrono::milliseconds g_refresh_rate = 100ms;

  bool exit = false;
  std::optional<ProgressReport> progress_report;

  while (!exit) {
    std::unique_lock<std::mutex> lock(m_statusline_mutex);
    if (!m_statusline_cv.wait_for(lock, g_refresh_rate,
                                  [&]() { return m_statusline_thread_exit; })) {
      // We hit the timeout. First check if we're asked to exit.
      if (m_statusline_thread_exit) {
        exit = true;
        continue;
      }

      StreamString stream;
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      SymbolContext symbol_ctx;
      if (auto frame_sp = exe_ctx.GetFrameSP())
        symbol_ctx = frame_sp->GetSymbolContext(eSymbolContextEverything);

      // Add the user-configured components.
      bool add_separator = false;
      for (const FormatEntity::Entry& entry : m_debugger.GetStatuslineFormat()) {
        if (add_separator)
          stream << " | ";
        add_separator =
            FormatEntity::Format(entry, stream, &symbol_ctx, &exe_ctx, nullptr,
                                 nullptr, false, false);
      }

      // Add progress reports at the end, if enabled.
      if (m_debugger.GetShowProgress()) {
        if (m_progress_reports.empty()) {
          progress_report.reset();
        } else {
          progress_report.emplace(m_progress_reports.back());
        }
        if (progress_report) {
          if (add_separator)
            stream << " | ";
          stream << progress_report->message;
        }
      }

      Draw(stream.GetString());
    } else {
      // We got notified and the predicate passed. First check if we're asked to
      // exit.
      if (m_statusline_thread_exit) {
        exit = true;
        continue;
      }
    }
  }

  return {};
}

bool Statusline::StartStatuslineThread() {
  Enable();
  if (!m_statusline_thread.IsJoinable()) {
    m_statusline_thread_exit = false;
    llvm::Expected<HostThread> statusline_thread = ThreadLauncher::LaunchThread(
        "lldb.debugger.statusline", [this] { return StatuslineThread(); });

    if (statusline_thread) {
      m_statusline_thread = *statusline_thread;
    } else {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Host), statusline_thread.takeError(),
                     "failed to launch host thread: {0}");
    }
  }
  return m_statusline_thread.IsJoinable();
}

void Statusline::StopStatuslineThread() {
  if (m_statusline_thread.IsJoinable()) {
    {
      std::lock_guard<std::mutex> guard(m_statusline_mutex);
      m_statusline_thread_exit = true;
    }
    m_statusline_cv.notify_one();
    m_statusline_thread.Join(nullptr);
  }
  Disable();
}

void Statusline::ReportProgress(const ProgressEventData &data) {
  // Make a local copy of the incoming progress report, which might get modified
  // below.
  ProgressReport progress_report{
      data.GetID(), data.IsFinite()
                        ? llvm::formatv("[{0}/{1}] {2}", data.GetCompleted(),
                                        data.GetTotal(), data.GetMessage())
                              .str()
                        : data.GetMessage()};

  std::lock_guard<std::mutex> guard(m_statusline_mutex);

  // Do some bookkeeping regardless of whether we're going to display
  // progress reports.
  auto it = std::find_if(
      m_progress_reports.begin(), m_progress_reports.end(),
      [&](const auto &report) { return report.id == progress_report.id; });
  if (it != m_progress_reports.end()) {
    const bool complete = data.GetCompleted() == data.GetTotal();
    if (complete)
      m_progress_reports.erase(it);
    else
      *it = progress_report;
  } else {
    m_progress_reports.push_back(progress_report);
  }
}
