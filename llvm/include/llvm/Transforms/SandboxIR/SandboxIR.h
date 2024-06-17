//===- SandboxIR.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sandbox IR is a lightweight overlay transactional IR on top of LLVM IR.
// Features:
// - You can save/rollback the state of the IR at any time.
// - Any changes made to Sandbox IR will automatically update the underlying
//   LLVM IR so both IRs are always in sync.
// - Feels like LLVM IR, similar API.
//
// SandboxIR forms a class hierarcy that resembles that of LLVM IR:
//
//          +- SBArgument   +- SBConstant     +- SBOpaqueInstruction
//          |               |                 |
// SBValue -+- SBUser ------+- SBInstruction -+- SBInsertElementInstruction
//          |                                 |
//          +- SBBasicBlock                   +- SBExtractElementInstruction
//          |                                 |
//          +- SBFunction                     +- SBShuffleVectorInstruction
//                                            |
//                                            +- SBStoreInstruction
//                                            |
//                                            +- SBLoadInstruction
//                                            |
//                                            +- SBCmpInstruction
//                                            |
//                                            +- SBCastInstruction
//                                            |
//                                            +- SBPHINode
//                                            |
//                                            +- SBSelectInstruction
//                                            |
//                                            +- SBBinaryOperator
//                                            |
//                                            +- SBUnaryOperator
//
// SBUse
//
#ifndef LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
#define LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H

#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class SBContext;

/// A SBValue has users. This is the base class.
class SBValue {
public:
  enum class ClassID : unsigned {
#define DEF_VALUE(ID, CLASS) ID,
#define DEF_USER(ID, CLASS) ID,
#define DEF_INSTR(ID, OPC, CLASS) ID,
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
  };

protected:
  static const char *getSubclassIDStr(ClassID ID) {
    switch (ID) {
      // clang-format off
#define DEF_VALUE(ID, CLASS) case ClassID::ID: return #ID;
#define DEF_USER(ID,  CLASS) case ClassID::ID: return #ID;
#define DEF_INSTR(ID, OPC, CLASS) case ClassID::ID: return #ID;
      // clang-format on
#include "llvm/Transforms/SandboxIR/SandboxIRValues.def"
    }
    llvm_unreachable("Unimplemented ID");
  }

  /// For isa/dyn_cast.
  ClassID SubclassID;
#ifndef NDEBUG
  /// A unique ID used for forming the name (used for debugging).
  unsigned UID;
#endif
  /// The LLVM Value that corresponds to this SBValue.
  /// NOTE: Some SBInstructions, like Packs, may include more than one value.
  Value *Val = nullptr;
  friend class ValueAttorney; // For Val

  /// All values point to the context.
  SBContext &Ctxt;
  // This is used by eraseFromParent().
  void clearValue() { Val = nullptr; }
  template <typename ItTy, typename SBTy> friend class LLVMOpUserItToSBTy;

public:
  SBValue(ClassID SubclassID, Value *Val, SBContext &Ctxt);
  virtual ~SBValue() = default;
  ClassID getSubclassID() const { return SubclassID; }

  Type *getType() const { return Val->getType(); }

  SBContext &getContext() const;
  virtual hash_code hashCommon() const {
    return hash_combine(SubclassID, &Ctxt, Val);
  }
  virtual hash_code hash() const = 0;
  friend hash_code hash_value(const SBValue &SBV) { return SBV.hash(); }
#ifndef NDEBUG
  /// Should crash if there is something wrong with the instruction.
  virtual void verify() const = 0;
  /// Returns the name in the form 'SB<number>.' like 'SB1.'
  std::string getName() const;
  virtual void dumpCommonHeader(raw_ostream &OS) const;
  void dumpCommonFooter(raw_ostream &OS) const;
  void dumpCommonPrefix(raw_ostream &OS) const;
  void dumpCommonSuffix(raw_ostream &OS) const;
  void printAsOperandCommon(raw_ostream &OS) const;
  friend raw_ostream &operator<<(raw_ostream &OS, const SBValue &SBV) {
    SBV.dump(OS);
    return OS;
  }
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  virtual void dumpVerbose(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dumpVerbose() const = 0;
#endif
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SANDBOXIR_SANDBOXIR_H
