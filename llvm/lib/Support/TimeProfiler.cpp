//===-- TimeProfiler.cpp - Hierarchical Time Profiler ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hierarchical time profiler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeProfiler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace llvm;

namespace {

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::time_point_cast;

struct TimeTraceProfilerInstances {
  std::mutex Lock;
  std::vector<TimeTraceProfiler *> List;
};

TimeTraceProfilerInstances &getTimeTraceProfilerInstances() {
  static TimeTraceProfilerInstances Instances;
  return Instances;
}

} // anonymous namespace

// Per Thread instance
static LLVM_THREAD_LOCAL TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;

TimeTraceProfiler *llvm::getTimeTraceProfilerInstance() {
  return TimeTraceProfilerInstance;
}

namespace {

using ClockType = steady_clock;
using TimePointType = time_point<ClockType>;
using DurationType = duration<ClockType::rep, ClockType::period>;
using CountAndDurationType = std::pair<size_t, DurationType>;
using NameAndCountAndDurationType =
    std::pair<std::string, CountAndDurationType>;

} // anonymous namespace

/// Represents an entry to be captured.
struct llvm::TimeTraceProfilerEntry {
  const TimePointType Start;
  const std::string Name;
  TimeTraceMetadata Metadata;

  TimeTraceProfilerEntry(TimePointType &&S, std::string &&N, std::string &&Dt)
      : Start(std::move(S)), Name(std::move(N)), Metadata() {
    Metadata.Detail = std::move(Dt);
  }

  TimeTraceProfilerEntry(TimePointType &&S, std::string &&N,
                         TimeTraceMetadata &&Mt)
      : Start(std::move(S)), Name(std::move(N)), Metadata(std::move(Mt)) {}

  // Calculate timings for FlameGraph. Cast time points to microsecond precision
  // rather than casting duration. This avoids truncation issues causing inner
  // scopes overruning outer scopes.
  ClockType::rep getFlameGraphStartUs(TimePointType StartTime) const {
    return (time_point_cast<microseconds>(Start) -
            time_point_cast<microseconds>(StartTime))
        .count();
  }

  virtual void writeEvent(json::OStream &J, sys::Process::Pid Pid, uint64_t Tid,
                          TimePointType StartTime) const {
    J.object([&] {
      J.attribute("pid", Pid);
      J.attribute("tid", int64_t(Tid));
      J.attribute("ts", getFlameGraphStartUs(StartTime));
      writeAdditionalEventFileds(J);
      J.attribute("name", Name);
      if (!Metadata.isEmpty()) {
        J.attributeObject("args", [&] {
          if (!Metadata.Detail.empty())
            J.attribute("detail", Metadata.Detail);
          if (!Metadata.File.empty())
            J.attribute("file", Metadata.File);
          if (Metadata.Line > 0)
            J.attribute("line", Metadata.Line);
        });
      }
    });
  }

  virtual void writeAdditionalEventFileds(json::OStream &J) const {}

  virtual ~TimeTraceProfilerEntry() = default;
};

struct llvm::InstantEvent : llvm::TimeTraceProfilerEntry {
  InstantEvent(TimePointType &&S, std::string &&N, std::string &&Dt)
      : TimeTraceProfilerEntry(std::move(S), std::move(N), std::move(Dt)) {}

  InstantEvent(TimePointType &&S, std::string &&N, TimeTraceMetadata &&Mt)
      : TimeTraceProfilerEntry(std::move(S), std::move(N), std::move(Mt)) {}

  void writeAdditionalEventFileds(json::OStream &J) const override {
    J.attribute("ph", "i");
  }
};

struct llvm::DurableEvent : llvm::TimeTraceProfilerEntry {
  DurableEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
               std::string &&Dt)
      : TimeTraceProfilerEntry(std::move(S), std::move(N), std::move(Dt)),
        InstantEvents{}, End(std::move(E)) {}

  DurableEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
               TimeTraceMetadata &&Mt)
      : TimeTraceProfilerEntry(std::move(S), std::move(N), std::move(Mt)),
        InstantEvents{}, End(std::move(E)) {}

  virtual void writeEvent(json::OStream &J, sys::Process::Pid Pid, uint64_t Tid,
                          TimePointType StartTime) const override {
    TimeTraceProfilerEntry::writeEvent(J, Pid, Tid, StartTime);
    writeInstantEvents(J, Pid, Tid, StartTime);
  }

  void writeInstantEvents(json::OStream &J, sys::Process::Pid Pid, uint64_t Tid,
                          TimePointType StartTime) const {
    for (auto IE : InstantEvents) {
      IE->writeEvent(J, Pid, Tid, StartTime);
    }
  }

  virtual void writeAdditionalEventFileds(json::OStream &J) const override {}

  ClockType::rep getFlameGraphDurUs() const {
    return (time_point_cast<microseconds>(End) -
            time_point_cast<microseconds>(Start))
        .count();
  }

  std::vector<std::shared_ptr<InstantEvent>> InstantEvents;
  TimePointType End;
};

struct llvm::CompleteEvent : llvm::DurableEvent {
  CompleteEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
                std::string &&Dt)
      : DurableEvent(std::move(S), std::move(E), std::move(N), std::move(Dt)) {}

  CompleteEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
                TimeTraceMetadata &&Mt)
      : DurableEvent(std::move(S), std::move(E), std::move(N), std::move(Mt)) {}

  virtual void writeAdditionalEventFileds(json::OStream &J) const override {
    J.attribute("ph", "X");
    J.attribute("dur", getFlameGraphDurUs());
  }
};

struct llvm::AsyncEvent : llvm::DurableEvent {
  AsyncEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
             std::string &&Dt)
      : DurableEvent(std::move(S), std::move(E), std::move(N), std::move(Dt)) {}

  AsyncEvent(TimePointType &&S, TimePointType &&E, std::string &&N,
             TimeTraceMetadata &&Mt)
      : DurableEvent(std::move(S), std::move(E), std::move(N), std::move(Mt)) {}

  void writeEvent(json::OStream &J, sys::Process::Pid Pid, uint64_t Tid,
                  TimePointType StartTime) const override {
    DurableEvent::writeEvent(J, Pid, Tid, StartTime);
    writeEndEvent(J, Pid, Tid, StartTime);
  }

  void writeEndEvent(json::OStream &J, sys::Process::Pid Pid, uint64_t Tid,
                     TimePointType StartTime) const {
    J.object([&] {
      J.attribute("pid", Pid);
      J.attribute("tid", int64_t(Tid));
      J.attribute("ts", getFlameGraphStartUs(StartTime) + getFlameGraphDurUs());
      J.attribute("cat", Name);
      J.attribute("ph", "e");
      J.attribute("id", 0);
      J.attribute("name", Name);
    });
  }

  void writeAdditionalEventFileds(json::OStream &J) const override {
    J.attribute("cat", Name);
    J.attribute("ph", "b");
    J.attribute("id", 0);
  }
};

struct llvm::TimeTraceProfiler {
  TimeTraceProfiler(unsigned TimeTraceGranularity = 0, StringRef ProcName = "",
                    bool TimeTraceVerbose = false)
      : BeginningOfTime(system_clock::now()), StartTime(ClockType::now()),
        ProcName(ProcName), Pid(sys::Process::getProcessId()),
        Tid(llvm::get_threadid()), TimeTraceGranularity(TimeTraceGranularity),
        TimeTraceVerbose(TimeTraceVerbose) {
    llvm::get_thread_name(ThreadName);
  }
  std::shared_ptr<DurableEvent> begin(std::string Name,
                                      llvm::function_ref<std::string()> Detail,
                                      bool AsyncEventType) {
    if (AsyncEventType) {
      Stack.emplace_back(std::make_shared<AsyncEvent>(
          ClockType::now(), TimePointType(), std::move(Name), Detail()));
    } else {
      Stack.emplace_back(std::make_shared<CompleteEvent>(
          ClockType::now(), TimePointType(), std::move(Name), Detail()));
    }
    return Stack.back();
  }

  std::shared_ptr<DurableEvent>
  begin(std::string Name, llvm::function_ref<TimeTraceMetadata()> Metadata,
        bool AsyncEventType) {

    if (AsyncEventType) {
      Stack.emplace_back(std::make_shared<AsyncEvent>(
          ClockType::now(), TimePointType(), std::move(Name), Metadata()));
    } else {
      Stack.emplace_back(std::make_shared<CompleteEvent>(
          ClockType::now(), TimePointType(), std::move(Name), Metadata()));
    }
    return Stack.back();
  }

  void insert(std::string Name, llvm::function_ref<std::string()> Detail) {
    if (Stack.empty())
      return;

    Stack.back().get()->InstantEvents.emplace_back(
        std::make_shared<InstantEvent>(ClockType::now(), std::move(Name),
                                       Detail()));
  }

  void insert(std::string Name,
              llvm::function_ref<TimeTraceMetadata()> Metadata) {
    if (Stack.empty())
      return;

    Stack.back().get()->InstantEvents.emplace_back(
        std::make_shared<InstantEvent>(ClockType::now(), std::move(Name),
                                       Metadata()));
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    end(Stack.back());
  }

  void end(std::shared_ptr<DurableEvent> &E) {
    assert(!Stack.empty() && "Must call begin() first");
    E->End = ClockType::now();

    // Calculate duration at full precision for overall counts.
    DurationType Duration = E->End - E->Start;

    // Only include events with a duration longer or equal to
    // TimeTraceGranularity msec.
    if (duration_cast<microseconds>(Duration).count() >= TimeTraceGranularity) {
      Entries.emplace_back(E);
    }

    // Track total time taken by each "name", but only the topmost levels of
    // them; e.g. if there's a template instantiation that instantiates other
    // templates from within, we only want to add the topmost one. "topmost"
    // happens to be the ones that don't have any currently open entries above
    // itself.
    if (llvm::none_of(llvm::drop_begin(llvm::reverse(Stack)),
                      [&](const std::shared_ptr<DurableEvent> &Val) {
                        return Val->Name == E->Name;
                      })) {
      auto &CountAndTotal = CountAndTotalPerName[E->Name];
      CountAndTotal.first++;
      CountAndTotal.second += Duration;
    };

    llvm::erase_if(Stack, [&](const std::shared_ptr<DurableEvent> &Val) {
      return Val.get() == E.get();
    });
  }

  // Write events from this TimeTraceProfilerInstance and
  // ThreadTimeTraceProfilerInstances.
  void write(raw_pwrite_stream &OS) {
    // Acquire Mutex as reading ThreadTimeTraceProfilerInstances.
    auto &Instances = getTimeTraceProfilerInstances();
    std::lock_guard<std::mutex> Lock(Instances.Lock);
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling write");
    assert(llvm::all_of(Instances.List,
                        [](const auto &TTP) { return TTP->Stack.empty(); }) &&
           "All profiler sections should be ended when calling write");

    json::OStream J(OS);
    J.objectBegin();
    J.attributeBegin("traceEvents");
    J.arrayBegin();

    // Emit all events for the main flame graph.
    for (const auto &E : Entries) {
      E->writeEvent(J, Pid, this->Tid, StartTime);
    }
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const auto &E : TTP->Entries) {
        E->writeEvent(J, Pid, TTP->Tid, StartTime);
      }

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    // Find highest used thread id.
    uint64_t MaxTid = this->Tid;
    for (const TimeTraceProfiler *TTP : Instances.List)
      MaxTid = std::max(MaxTid, TTP->Tid);

    // Combine all CountAndTotalPerName from threads into one.
    StringMap<CountAndDurationType> AllCountAndTotalPerName;
    auto combineStat = [&](const auto &Stat) {
      StringRef Key = Stat.getKey();
      auto Value = Stat.getValue();
      auto &CountAndTotal = AllCountAndTotalPerName[Key];
      CountAndTotal.first += Value.first;
      CountAndTotal.second += Value.second;
    };
    for (const auto &Stat : CountAndTotalPerName)
      combineStat(Stat);
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const auto &Stat : TTP->CountAndTotalPerName)
        combineStat(Stat);

    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(AllCountAndTotalPerName.size());
    for (const auto &Total : AllCountAndTotalPerName)
      SortedTotals.emplace_back(std::string(Total.getKey()), Total.getValue());

    llvm::sort(SortedTotals, [](const NameAndCountAndDurationType &A,
                                const NameAndCountAndDurationType &B) {
      return A.second.second > B.second.second;
    });

    // Report totals on separate threads of tracing file.
    uint64_t TotalTid = MaxTid + 1;
    for (const NameAndCountAndDurationType &Total : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(Total.second.second).count();
      auto Count = AllCountAndTotalPerName[Total.first].first;

      J.object([&] {
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(TotalTid));
        J.attribute("ph", "X");
        J.attribute("ts", 0);
        J.attribute("dur", DurUs);
        J.attribute("name", "Total " + Total.first);
        J.attributeObject("args", [&] {
          J.attribute("count", int64_t(Count));
          J.attribute("avg ms", int64_t(DurUs / Count / 1000));
        });
      });

      ++TotalTid;
    }

    auto writeMetadataEvent = [&](const char *Name, uint64_t Tid,
                                  StringRef arg) {
      J.object([&] {
        J.attribute("cat", "");
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(Tid));
        J.attribute("ts", 0);
        J.attribute("ph", "M");
        J.attribute("name", Name);
        J.attributeObject("args", [&] { J.attribute("name", arg); });
      });
    };

    writeMetadataEvent("process_name", Tid, ProcName);
    writeMetadataEvent("thread_name", Tid, ThreadName);
    for (const TimeTraceProfiler *TTP : Instances.List)
      writeMetadataEvent("thread_name", TTP->Tid, TTP->ThreadName);

    J.arrayEnd();
    J.attributeEnd();

    // Emit the absolute time when this TimeProfiler started.
    // This can be used to combine the profiling data from
    // multiple processes and preserve actual time intervals.
    J.attribute("beginningOfTime",
                time_point_cast<microseconds>(BeginningOfTime)
                    .time_since_epoch()
                    .count());

    J.objectEnd();
  }

  SmallVector<std::shared_ptr<DurableEvent>, 16> Stack;
  SmallVector<std::shared_ptr<DurableEvent>, 128> Entries;
  StringMap<CountAndDurationType> CountAndTotalPerName;
  // System clock time when the session was begun.
  const time_point<system_clock> BeginningOfTime;
  // Profiling clock time when the session was begun.
  const TimePointType StartTime;
  const std::string ProcName;
  const sys::Process::Pid Pid;
  SmallString<0> ThreadName;
  const uint64_t Tid;

  // Minimum time granularity (in microseconds)
  const unsigned TimeTraceGranularity;

  // Make time trace capture verbose event details (e.g. source filenames). This
  // can increase the size of the output by 2-3 times.
  const bool TimeTraceVerbose;
};

bool llvm::isTimeTraceVerbose() {
  return getTimeTraceProfilerInstance() &&
         getTimeTraceProfilerInstance()->TimeTraceVerbose;
}

void llvm::timeTraceProfilerInitialize(unsigned TimeTraceGranularity,
                                       StringRef ProcName,
                                       bool TimeTraceVerbose) {
  assert(TimeTraceProfilerInstance == nullptr &&
         "Profiler should not be initialized");
  TimeTraceProfilerInstance = new TimeTraceProfiler(
      TimeTraceGranularity, llvm::sys::path::filename(ProcName),
      TimeTraceVerbose);
}

// Removes all TimeTraceProfilerInstances.
// Called from main thread.
void llvm::timeTraceProfilerCleanup() {
  delete TimeTraceProfilerInstance;
  TimeTraceProfilerInstance = nullptr;

  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  for (auto *TTP : Instances.List)
    delete TTP;
  Instances.List.clear();
}

// Finish TimeTraceProfilerInstance on a worker thread.
// This doesn't remove the instance, just moves the pointer to global vector.
void llvm::timeTraceProfilerFinishThread() {
  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  Instances.List.push_back(TimeTraceProfilerInstance);
  TimeTraceProfilerInstance = nullptr;
}

void llvm::timeTraceProfilerWrite(raw_pwrite_stream &OS) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");
  TimeTraceProfilerInstance->write(OS);
}

Error llvm::timeTraceProfilerWrite(StringRef PreferredFileName,
                                   StringRef FallbackFileName) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");

  std::string Path = PreferredFileName.str();
  if (Path.empty()) {
    Path = FallbackFileName == "-" ? "out" : FallbackFileName.str();
    Path += ".time-trace";
  }

  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_TextWithCRLF);
  if (EC)
    return createStringError(EC, "Could not open " + Path);

  timeTraceProfilerWrite(OS);
  return Error::success();
}

std::shared_ptr<DurableEvent> llvm::timeTraceProfilerBegin(StringRef Name,
                                                           StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(
        std::string(Name), [&]() { return std::string(Detail); }, false);
  return nullptr;
}

std::shared_ptr<DurableEvent>
llvm::timeTraceProfilerBegin(StringRef Name,
                             llvm::function_ref<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(std::string(Name), Detail, false);
  return nullptr;
}

std::shared_ptr<DurableEvent>
llvm::timeTraceProfilerBegin(StringRef Name,
                             llvm::function_ref<TimeTraceMetadata()> Metadata) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(std::string(Name), Metadata, false);
  return nullptr;
}

void llvm::timeTraceProfilerInsert(StringRef Name, StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->insert(std::string(Name),
                                      [&]() { return std::string(Detail); });
}

void llvm::timeTraceProfilerInsert(
    StringRef Name, llvm::function_ref<TimeTraceMetadata()> Metadata) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->insert(std::string(Name), Metadata);
}

std::shared_ptr<DurableEvent>
llvm::timeTraceAsyncProfilerBegin(StringRef Name, StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return TimeTraceProfilerInstance->begin(
        std::string(Name), [&]() { return std::string(Detail); }, true);
  return nullptr;
}

void llvm::timeTraceProfilerEnd() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end();
}

void llvm::timeTraceProfilerEnd(std::shared_ptr<DurableEvent> &E) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end(E);
}
