//===- llvm/TableGen/TGTimer.h - Class for TableGen Timer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TableGen timer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_TGTIMER_H
#define LLVM_TABLEGEN_TGTIMER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"

namespace llvm {

// Timer related functionality f or TableGen backends.
class TGTimer {
private:
  TimerGroup *TimingGroup = nullptr;
  Timer *LastTimer = nullptr;
  bool BackendTimer = false; // Is last timer special backend overall timer?

public:
  TGTimer() = default;

  /// Start phase timing; called if the --time-phases option is specified.
  void startPhaseTiming() {
    TimingGroup = new TimerGroup("TableGen", "TableGen Phase Timing");
  }

  /// Start timing a phase. Automatically stops any previous phase timer.
  void startTimer(StringRef Name);

  /// Stop timing a phase.
  void stopTimer();

  /// Start timing the overall backend. If the backend itself starts a timer,
  /// then this timer is cleared.
  void startBackendTimer(StringRef Name);

  /// Stop timing the overall backend.
  void stopBackendTimer();

  /// Stop phase timing and print the report.
  void stopPhaseTiming() {
    delete TimingGroup;
    TimingGroup = nullptr;
  }
};

} // end namespace llvm

#endif // LLVM_TABLEGEN_TGTIMER_H
