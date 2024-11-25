//===- BPSectionOrderer.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file uses Balanced Partitioning to order sections to improve startup
/// time and compressed size.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_BPSECTION_ORDERER_H
#define LLD_ELF_BPSECTION_ORDERER_H

#include "InputFiles.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Symbols.h"
#include "lld/Common/BPSectionOrdererBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/xxhash.h"

namespace lld::elf {

class InputSection;

class BPSymbolELF : public BPSymbol {
  const Symbol *sym;

public:
  explicit BPSymbolELF(const Symbol *s) : sym(s) {}

  llvm::StringRef getName() const override { return sym->getName(); }

  const Defined *asDefined() const {
    return llvm::dyn_cast_or_null<Defined>(sym);
  }

  BPSymbol *asDefinedSymbol() override { return asDefined() ? this : nullptr; }

  std::optional<uint64_t> getValue() const override {
    if (auto *d = asDefined())
      return d->value;
    return {};
  }

  std::optional<uint64_t> getSize() const override {
    if (auto *d = asDefined())
      return d->size;
    return {};
  }

  InputSectionBase *getInputSection() const {
    if (auto *d = llvm::dyn_cast<Defined>(sym))
      return llvm::dyn_cast_or_null<InputSectionBase>(d->section);
    return nullptr;
  }

  const Symbol *getSymbol() const { return sym; }
};

class BPSectionELF : public BPSectionBase {
  const InputSectionBase *isec;
  std::unique_ptr<BPSymbolELF> symbol;

public:
  explicit BPSectionELF(const InputSectionBase *sec,
                        std::unique_ptr<BPSymbolELF> sym)
      : isec(sec), symbol(std::move(sym)) {}

  const InputSectionBase *getSection() const { return isec; }

  BPSymbolELF *getSymbol() const { return symbol.get(); }
  llvm::StringRef getName() const override { return isec->name; }

  uint64_t getSize() const override { return isec->getSize(); }

  bool isCodeSection() const override {
    return isec->flags & llvm::ELF::SHF_EXECINSTR;
  }

  bool hasValidData() const override {
    return isec && !isec->content().empty();
  }

  llvm::ArrayRef<uint8_t> getSectionData() const override {
    return isec->content();
  }

  llvm::ArrayRef<std::unique_ptr<BPSymbol>> getSymbols() const override {
    return llvm::ArrayRef<std::unique_ptr<BPSymbol>>(
        reinterpret_cast<const std::unique_ptr<BPSymbol> *>(&symbol), 1);
  }

  bool needResolveLinkageName(llvm::StringRef &name) const override {
    return false;
  }

  void getSectionHash(llvm::SmallVectorImpl<uint64_t> &hashes) const override {
    constexpr unsigned windowSize = 4;

    // Calculate content hashes
    size_t size = isec->content().size();
    for (size_t i = 0; i < size; i++) {
      auto window = isec->content().drop_front(i).take_front(windowSize);
      hashes.push_back(xxHash64(window));
    }

    // TODO: Calculate relocation hashes.
    // Since in ELF, relocations are complex, but the effect without them are
    // good enough, we just use 0 as their hash.

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  static bool classof(const BPSectionBase *s) { return true; }
};

/// Run Balanced Partitioning to find the optimal function and data order to
/// improve startup time and compressed size.
///
/// It is important that -ffunction-sections and -fdata-sections are used to
/// ensure functions and data are in their own sections and thus can be
/// reordered.
llvm::DenseMap<const lld::elf::InputSectionBase *, int>
runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                        bool forFunctionCompression, bool forDataCompression,
                        bool compressionSortStartupFunctions, bool verbose);
} // namespace lld::elf

#endif
