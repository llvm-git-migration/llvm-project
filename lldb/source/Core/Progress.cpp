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
  std::lock_guard<std::mutex> guard(m_mutex);
  ReportProgress();
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_completed)
    m_completed = m_total;
  ReportProgress();
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
                             m_debugger_id);
  }
}

void ProgressManager::Initialize() {
  lldbassert(!InstanceImpl() && "A progress report manager already exists.");
  InstanceImpl().emplace();
}

void ProgressManager::Terminate() {
  lldbassert(InstanceImpl() &&
             "A progress report manager has already been terminated.");
  InstanceImpl().reset();
}

std::optional<ProgressManager> &ProgressManager::InstanceImpl() {
  static std::optional<ProgressManager> g_progress_manager;
  return g_progress_manager;
}

ProgressManager::ProgressManager() : m_progress_map() {}

ProgressManager::~ProgressManager() {}

ProgressManager &ProgressManager::Instance() { return *InstanceImpl(); }

void ProgressManager::Increment(std::string title) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pair = m_progress_map.insert(std::pair(title, 1));

  // If pair.first is not empty after insertion it means that that
  // category was entered for the first time and should not be incremented
  if (!pair.first->first().empty())
    ++pair.first->second;
}

void ProgressManager::Decrement(std::string title) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pos = m_progress_map.find(title);

  if (pos == m_progress_map.end())
    return;

  // Remove the category from the map if the refcount reaches 0
  if (--pos->second == 0)
    m_progress_map.erase(title);
}
