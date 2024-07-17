//===- Tracker.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Tracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <sstream>

using namespace llvm::sandboxir;

IRChangeBase::IRChangeBase(TrackID ID, Tracker &Parent)
    : ID(ID), Parent(Parent) {
#ifndef NDEBUG
  Idx = Parent.size();

  assert(!Parent.InMiddleOfCreatingChange &&
         "We are in the middle of creating another change!");
  if (Parent.isTracking())
    Parent.InMiddleOfCreatingChange = true;
#endif // NDEBUG
}

#ifndef NDEBUG
void UseSet::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Tracker::~Tracker() {
  assert(Changes.empty() && "You must accept or revert changes!");
}

void Tracker::track(std::unique_ptr<IRChangeBase> &&Change) {
  assert(State != TrackerState::Revert &&
         "No changes should be tracked during revert()!");
  Changes.push_back(std::move(Change));

#ifndef NDEBUG
  InMiddleOfCreatingChange = false;
#endif
}

void Tracker::save() { State = TrackerState::Record; }

void Tracker::revert() {
  assert(State == TrackerState::Record && "Forgot to save()!");
  State = TrackerState::Revert;
  for (auto &Change : reverse(Changes))
    Change->revert();
  Changes.clear();
  State = TrackerState::Disabled;
}

void Tracker::accept() {
  assert(State == TrackerState::Record && "Forgot to save()!");
  State = TrackerState::Accept;
  for (auto &Change : Changes)
    Change->accept();
  Changes.clear();
  State = TrackerState::Disabled;
}

#ifndef NDEBUG
void Tracker::dump(raw_ostream &OS) const {
  for (const auto &ChangePtr : Changes) {
    ChangePtr->dump(OS);
    OS << "\n";
  }
}
void Tracker::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
