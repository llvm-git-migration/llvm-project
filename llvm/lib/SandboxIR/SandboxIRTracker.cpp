//===- SandboxIRTracker.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIRTracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include <sstream>

using namespace llvm::sandboxir;

IRChangeBase::IRChangeBase(TrackID ID, SandboxIRTracker &Parent)
    : ID(ID), Parent(Parent) {
#ifndef NDEBUG
  Idx = Parent.size();

  assert(!Parent.InMiddleOfCreatingChange &&
         "We are in the middle of creating another change!");
  if (Parent.tracking())
    Parent.InMiddleOfCreatingChange = true;
#endif // NDEBUG
}

#ifndef NDEBUG
void UseSet::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

SandboxIRTracker::~SandboxIRTracker() {
  assert(Changes.empty() && "You must accept or revert changes!");
}

void SandboxIRTracker::track(std::unique_ptr<IRChangeBase> &&Change) {
#ifndef NDEBUG
  assert(State != TrackerState::Revert &&
         "No changes should be tracked during revert()!");
#endif // NDEBUG
  Changes.push_back(std::move(Change));

#ifndef NDEBUG
  InMiddleOfCreatingChange = false;
#endif
}

void SandboxIRTracker::save() { State = TrackerState::Record; }

void SandboxIRTracker::revert() {
  auto SavedState = State;
  State = TrackerState::Revert;
  for (auto &Change : reverse(Changes))
    Change->revert();
  Changes.clear();
  State = SavedState;
}

void SandboxIRTracker::accept() {
  auto SavedState = State;
  State = TrackerState::Accept;
  for (auto &Change : Changes)
    Change->accept();
  Changes.clear();
  State = SavedState;
}

#ifndef NDEBUG
void SandboxIRTracker::dump(raw_ostream &OS) const {
  for (const auto &ChangePtr : Changes) {
    ChangePtr->dump(OS);
    OS << "\n";
  }
}
void SandboxIRTracker::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
