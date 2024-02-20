//===-- Progress.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Progress.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Utility/StreamString.h"

#include <mutex>
#include <optional>

using namespace lldb;
using namespace lldb_private;

std::atomic<uint64_t> Progress::g_id(0);

Progress::Progress(std::string title, std::string details,
                   std::optional<uint64_t> total,
                   lldb_private::Debugger *debugger)
    : m_title(title), m_details(details), m_id(++g_id), m_completed(0),
      m_total(Progress::kNonDeterministicTotal) {
  if (total)
    m_total = *total;

  if (debugger)
    m_debugger_id = debugger->GetID();

  m_progress_data = {m_title,
                     m_details,
                     m_id,
                     m_completed,
                     m_total,
                     m_debugger_id,
                     Debugger::eBroadcastBitProgress};
  std::lock_guard<std::mutex> guard(m_mutex);
  ProgressManager::Instance().Increment(m_progress_data);
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_completed)
    m_completed = m_total;
  m_progress_data.completed = m_completed;
  ProgressManager::Instance().Decrement(m_progress_data);
}

void Progress::Increment(uint64_t amount,
                         std::optional<std::string> updated_detail) {
  if (amount > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (updated_detail)
      m_details = std::move(updated_detail.value());
    // Watch out for unsigned overflow and make sure we don't increment too
    // much and exceed m_total.
    if (m_total && (amount > (m_total - m_completed)))
      m_completed = m_total;
    else
      m_completed += amount;
    ReportProgress();
  }
}

void Progress::ReportProgress() {
  if (!m_complete) {
    // Make sure we only send one notification that indicates the progress is
    // complete
    m_complete = m_completed == m_total;
    Debugger::ReportProgress(m_id, m_title, m_details, m_completed, m_total,
                             m_debugger_id, Debugger::eBroadcastBitProgress);
  }
}

ProgressManager::ProgressManager() : m_progress_category_map() {}

ProgressManager::~ProgressManager() {}

ProgressManager &ProgressManager::Instance() {
  static std::once_flag g_once_flag;
  static ProgressManager *g_progress_manager = nullptr;
  std::call_once(g_once_flag, []() {
    // NOTE: known leak to avoid global destructor chain issues.
    g_progress_manager = new ProgressManager();
  });
  return *g_progress_manager;
}

void ProgressManager::Increment(Progress::ProgressData progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  progress_data.progress_broadcast_bit =
      m_progress_category_map.contains(progress_data.title)
          ? Debugger::eBroadcastBitProgress
          : Debugger::eBroadcastBitProgressCategory;
  ReportProgress(progress_data);
  m_progress_category_map[progress_data.title].first++;
}

void ProgressManager::Decrement(Progress::ProgressData progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pos = m_progress_category_map.find(progress_data.title);

  if (pos == m_progress_category_map.end())
    return;

  if (pos->second.first <= 1) {
    progress_data.progress_broadcast_bit =
        Debugger::eBroadcastBitProgressCategory;
    m_progress_category_map.erase(progress_data.title);
  } else
    --pos->second.first;

  ReportProgress(progress_data);
}

void ProgressManager::ReportProgress(Progress::ProgressData progress_data) {
  Debugger::ReportProgress(progress_data.progress_id, progress_data.title,
                           progress_data.details, progress_data.completed,
                           progress_data.total, progress_data.debugger_id,
                           progress_data.progress_broadcast_bit);
}
