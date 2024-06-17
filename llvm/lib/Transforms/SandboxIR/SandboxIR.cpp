//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/SandboxIR/SandboxIR.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm;

SBValue::SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt)
    : SubclassID(SubclassID), Val(Val), Ctxt(Ctxt) {
#ifndef NDEBUG
  UID = 0; // FIXME: Once SBContext is available.
#endif
}

#ifndef NDEBUG
std::string SBValue::getName() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void SBValue::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void SBValue::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";
}

void SBValue::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void SBValue::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void SBValue::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void SBValue::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
