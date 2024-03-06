//===-- Alarm.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_ALARM_H
#define LLDB_HOST_ALARM_H

#include "lldb/Host/HostThread.h"
#include "lldb/lldb-types.h"
#include "llvm/Support/Chrono.h"

namespace lldb_private {

class Alarm {
public:
  using Handle = uint64_t;
  using Callback = std::function<void()>;
  using TimePoint = llvm::sys::TimePoint<>;
  using Duration = std::chrono::milliseconds;

  Alarm(Duration timeout, bool run_callback_on_exit = false);
  ~Alarm();

  Handle Create(Callback callback);
  bool Restart(Handle handle);
  bool Cancel(Handle handle);

  static constexpr Handle INVALID_HANDLE = 0;

private:
  /// Helper functions to start, stop and check the status of the alarm thread.
  /// @{
  void StartAlarmThread();
  void StopAlarmThread();
  bool AlarmThreadRunning();
  /// @}

  /// Return an unique, monotonically increasing handle.
  static Handle GetNextUniqueHandle();

  TimePoint GetNextExpiration() const;

  /// Alarm entry.
  struct Entry {
    Handle handle;
    Callback callback;
    TimePoint expiration;

    Entry(Callback callback, TimePoint expiration);
    bool operator==(const Entry &rhs) { return handle == rhs.handle; }
  };

  /// List of alarm entries.
  std::vector<Entry> m_entries;

  /// Timeout between when an alarm is created and when it fires.
  Duration m_timeout;

  /// The alarm thread.
  /// @{
  HostThread m_alarm_thread;
  lldb::thread_result_t AlarmThread();
  /// @}

  /// Synchronize access between the alarm thread and the main thread.
  std::mutex m_alarm_mutex;

  /// Condition variable used to wake up the alarm thread.
  std::condition_variable m_alarm_cv;

  /// Flag to signal the alarm thread that something changed and we need to
  // recompute the next alarm.
  bool m_recompute_next_alarm = false;

  /// Flag to signal the alarm thread to exit.
  bool m_exit = false;

  /// Flag to signal we should run all callbacks on exit.
  bool m_run_callbacks_on_exit = false;
};

} // namespace lldb_private

#endif // LLDB_HOST_ALARM_H
