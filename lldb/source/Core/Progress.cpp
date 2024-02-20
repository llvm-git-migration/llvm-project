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

#include <cstdint>
#include <mutex>
#include <optional>

using namespace lldb;
using namespace lldb_private;

std::atomic<uint64_t> Progress::g_id(0);

Progress::Progress(std::string title, std::string details,
                   std::optional<uint64_t> total,
                   lldb_private::Debugger *debugger)
    : m_progress_data{title,
                      details,
                      ++g_id,
                      /*m_progress_data.completed*/ 0,
                      Progress::kNonDeterministicTotal,
                      /*m_progress_data.debugger_id*/ std::nullopt} {
  if (total)
    m_progress_data.total = *total;

  if (debugger)
    m_progress_data.debugger_id = debugger->GetID();

  std::lock_guard<std::mutex> guard(m_mutex);
  ReportProgress();
  ProgressManager::Instance().Increment(m_progress_data);
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_progress_data.completed)
    m_progress_data.completed = m_progress_data.total;
  m_progress_data.completed = m_progress_data.completed;
  ReportProgress();
  ProgressManager::Instance().Decrement(m_progress_data);
}

void Progress::Increment(uint64_t amount,
                         std::optional<std::string> updated_detail) {
  if (amount > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (updated_detail)
      m_progress_data.details = std::move(updated_detail.value());
    // Watch out for unsigned overflow and make sure we don't increment too
    // much and exceed the total.
    if (m_progress_data.total &&
        (amount > (m_progress_data.total - m_progress_data.completed)))
      m_progress_data.completed = m_progress_data.total;
    else
      m_progress_data.completed += amount;
    ReportProgress();
  }
}

void Progress::ReportProgress() {
  if (!m_complete) {
    // Make sure we only send one notification that indicates the progress is
    // complete
    m_complete = m_progress_data.completed == m_progress_data.total;
    Debugger::ReportProgress(m_progress_data.progress_id, m_progress_data.title,
                             m_progress_data.details, m_progress_data.completed,
                             m_progress_data.total,
                             m_progress_data.debugger_id);
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

void ProgressManager::Increment(const Progress::ProgressData &progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  // If the current category exists in the map then it is not an initial report,
  // therefore don't broadcast to the category bit.
  if (!m_progress_category_map.contains(progress_data.title))
    ReportProgress(progress_data);
  m_progress_category_map[progress_data.title].first++;
}

void ProgressManager::Decrement(const Progress::ProgressData &progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pos = m_progress_category_map.find(progress_data.title);

  if (pos == m_progress_category_map.end())
    return;

  if (pos->second.first <= 1) {
    m_progress_category_map.erase(progress_data.title);
    ReportProgress(progress_data);
  } else {
    --pos->second.first;
  }
}

void ProgressManager::ReportProgress(Progress::ProgressData progress_data) {
  // The category bit only keeps track of when progress report categories have
  // started and ended, so clear the details when broadcasting to it since that
  // bit doesn't need that information.
  progress_data.details = "";
  Debugger::ReportProgress(progress_data.progress_id, progress_data.title,
                           progress_data.details, progress_data.completed,
                           progress_data.total, progress_data.debugger_id,
                           Debugger::eBroadcastBitProgressCategory);
}
