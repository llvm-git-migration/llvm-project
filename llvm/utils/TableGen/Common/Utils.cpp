//===- Utils.cpp - Common Utilities -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>

using namespace llvm;

/// Sort an array of Records on the "Name" field, and check for records with
/// duplicate "Name" field. If duplicates are found, report a fatal error.
void llvm::sortAndReportDuplicates(std::vector<Record *> &Records,
                                   StringRef ObjectName) {
  llvm::sort(Records, LessRecordFieldNameAndName());

  auto I = std::adjacent_find(Records.begin(), Records.end(),
                              [](const Record *Rec1, const Record *Rec2) {
                                return Rec1->getValueAsString("Name") ==
                                       Rec2->getValueAsString("Name");
                              });
  if (I == Records.end())
    return;

  // Found a duplicate name.
  const Record *First = *I;
  const Record *Second = *(I + 1);
  StringRef Name = First->getValueAsString("Name");
  PrintError(Second, ObjectName + " `" + Name + "` has multiple definitions.");
  PrintFatalNote(First, "Another definition here.");
}
