//===- Pass.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_PASS_H
#define LLVM_SANDBOXIR_PASS_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::sandboxir {

class Function;

/// The base class of a Sandbox IR Pass.
class Pass {
public:
  enum class ClassID : unsigned {
    FunctionPass,
  };
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
    case ClassID::FunctionPass:
      return "FunctionPass";
    }
    llvm_unreachable("Unimplemented ID");
  }

protected:
  /// The pass name.
  const std::string Name;
  /// The command-line flag used to specify that this pass should run.
  const std::string Flag;
  /// Used for isa/cast/dyn_cast.
  ClassID SubclassID;

public:
  Pass(StringRef Name, StringRef Flag, ClassID SubclassID)
      : Name(Name), Flag(Flag), SubclassID(SubclassID) {}
  virtual ~Pass() {}
  /// \Returns the name of the pass.
  StringRef getName() const { return Name; }
  /// \Returns the command-line flag used to enable the pass.
  StringRef getFlag() const { return Flag; }
  ClassID getSubclassID() const { return SubclassID; }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const Pass &Pass) {
    Pass.dump(OS);
    return OS;
  }
  void dump(raw_ostream &OS) const { OS << Name << " " << Flag; }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// A pass that runs on a sandbox::Function.
class FunctionPass : public Pass {
protected:
  FunctionPass(StringRef Name, StringRef Flag, ClassID PassID)
      : Pass(Name, Flag, PassID) {}

public:
  FunctionPass(StringRef Name, StringRef Flag)
      : Pass(Name, Flag, ClassID::FunctionPass) {}
  /// For isa/dyn_cast etc.
  static bool classof(const Pass *From) {
    switch (From->getSubclassID()) {
    case ClassID::FunctionPass:
      return true;
    }
  }
  /// \Returns true if it modifies \p F.
  virtual bool runOnFunction(Function &F) = 0;
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_PASS_H
