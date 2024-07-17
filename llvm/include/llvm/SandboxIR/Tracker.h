//===- Tracker.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the component of SandboxIR that tracks all changes made to its
// state, such that we can revert the state when needed.
//
// Tracking changes
// ----------------
// The user needs to call `Tracker::save()` to enable tracking changes
// made to SandboxIR. From that point on, any change made to SandboxIR, will
// automatically create a change tracking object and register it with the
// tracker. IR-change objects are subclasses of `IRChangeBase` and get
// registered with the `Tracker::track()` function. The change objects
// are saved in the order they are registered with the tracker and are stored in
// the `Tracker::Changes` vector. All of this is done transparently to
// the user.
//
// Reverting changes
// -----------------
// Calling `Tracker::revert()` will restore the state saved when
// `Tracker::save()` was called. Internally this goes through the
// change objects in `Tracker::Changes` in reverse order, calling their
// `IRChangeBase::revert()` function one by one.
//
// Accepting changes
// -----------------
// The user needs to either revert or accept changes before the tracker object
// is destroyed. This is enforced in the tracker's destructor.
// This is the job of `Tracker::accept()`. Internally this will go
// through the change objects in `Tracker::Changes` in order, calling
// `IRChangeBase::accept()`.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_TRACKER_H
#define LLVM_SANDBOXIR_TRACKER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/Use.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <regex>

namespace llvm::sandboxir {

class BasicBlock;

/// Each IR change type has an ID.
enum class TrackID {
  UseSet,
};

#ifndef NDEBUG
static const char *trackIDToStr(TrackID ID) {
  switch (ID) {
  case TrackID::UseSet:
    return "UseSet";
  }
  llvm_unreachable("Unimplemented ID");
}
#endif // NDEBUG

class Tracker;

/// The base class for IR Change classes.
class IRChangeBase {
protected:
#ifndef NDEBUG
  unsigned Idx = 0;
#endif
  const TrackID ID;
  Tracker &Parent;

public:
  IRChangeBase(TrackID ID, Tracker &Parent);
  TrackID getTrackID() const { return ID; }
  /// This runs when changes get reverted.
  virtual void revert() = 0;
  /// This runs when changes get accepted.
  virtual void accept() = 0;
  virtual ~IRChangeBase() = default;
#ifndef NDEBUG
  void dumpCommon(raw_ostream &OS) const {
    OS << Idx << ". " << trackIDToStr(ID);
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  friend raw_ostream &operator<<(raw_ostream &OS, const IRChangeBase &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

/// Tracks the change of the source Value of a sandboxir::Use.
class UseSet : public IRChangeBase {
  Use U;
  Value *OrigV = nullptr;

public:
  UseSet(const Use &U, Tracker &Tracker)
      : IRChangeBase(TrackID::UseSet, Tracker), U(U), OrigV(U.get()) {}
  // For isa<> etc.
  static bool classof(const IRChangeBase *Other) {
    return Other->getTrackID() == TrackID::UseSet;
  }
  void revert() final { U.set(OrigV); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { dumpCommon(OS); }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

/// The tracker collects all the change objects and implements the main API for
/// saving / reverting / accepting.
class Tracker {
public:
  enum class TrackerState {
    Disabled, ///> Tracking is disabled
    Record,   ///> Tracking changes
    Revert,   ///> Undoing changes
    Accept,   ///> Accepting changes
  };

private:
  /// The list of changes that are being tracked.
  SmallVector<std::unique_ptr<IRChangeBase>> Changes;
  /// The current state of the tracker.
  TrackerState State = TrackerState::Disabled;

public:
#ifndef NDEBUG
  /// Helps catch bugs where we are creating new change objects while in the
  /// middle of creating other change objects.
  bool InMiddleOfCreatingChange = false;
#endif // NDEBUG

  Tracker() = default;
  ~Tracker();
  /// Record \p Change and take ownership. This is the main function used to
  /// track Sandbox IR changes.
  void track(std::unique_ptr<IRChangeBase> &&Change);
  /// \Returns true if the tracker is recording changes.
  bool isTracking() const { return State == TrackerState::Record; }
  /// \Returns the current state of the tracker.
  TrackerState getState() const { return State; }
  /// Turns on IR tracking.
  void save();
  /// Stops tracking and accept changes.
  void accept();
  /// Stops tracking and reverts to saved state.
  void revert();
  /// \Returns the number of change entries recorded so far.
  unsigned size() const { return Changes.size(); }
  /// \Returns true if there are no change entries recorded so far.
  bool empty() const { return Changes.empty(); }

#ifndef NDEBUG
  /// \Returns the \p Idx'th change. This is used for testing.
  IRChangeBase *getChange(unsigned Idx) const { return Changes[Idx].get(); }
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const Tracker &Tracker) {
    Tracker.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_TRACKER_H
