//===- OptEmitter.cpp - Helper for emitting options -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptEmitter.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/OptStrCmp.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

// Returns true if A is ordered before B.
// Note: Keep this in sync with `operator <` for OptTable::Info in OptTable.cpp.
bool llvm::IsOptionRecordsLess(const Record *A, const Record *B) {
  if (A == B)
    return false;

  // Sentinel options precede all others and are only ordered by precedence.
  const Record *AKind = A->getValueAsDef("Kind");
  const Record *BKind = B->getValueAsDef("Kind");

  bool ASent = AKind->getValueAsBit("Sentinel");
  bool BSent = BKind->getValueAsBit("Sentinel");
  if (ASent != BSent)
    return ASent;

  // Compare options by name, unless they are sentinels.
  if (!ASent) {
    if (int Cmp = opt::StrCmpOptionName(A->getValueAsString("Name"),
                                        B->getValueAsString("Name")))
      return Cmp < 0;

    std::vector<StringRef> APrefixes = A->getValueAsListOfStrings("Prefixes");
    std::vector<StringRef> BPrefixes = B->getValueAsListOfStrings("Prefixes");
    for (const auto &[APre, BPre] : zip(APrefixes, BPrefixes)) {
      if (int Cmp = opt::StrCmpOptionName(APre, BPre))
        return Cmp < 0;
    }
  }

  // Then by the kind precedence;
  int APrec = AKind->getValueAsInt("Precedence");
  int BPrec = BKind->getValueAsInt("Precedence");
  if (APrec == BPrec && A->getValueAsListOfStrings("Prefixes") ==
                            B->getValueAsListOfStrings("Prefixes")) {
    PrintError(A->getLoc(), Twine("Option is equivalent to"));
    PrintError(B->getLoc(), Twine("Other defined here"));
    PrintFatalError("Equivalent Options found.");
  }
  return APrec < BPrec;
}
