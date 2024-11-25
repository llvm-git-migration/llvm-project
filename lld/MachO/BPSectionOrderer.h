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

#ifndef LLD_MACHO_BPSECTION_ORDERER_H
#define LLD_MACHO_BPSECTION_ORDERER_H

#include "InputSection.h"
#include "Relocations.h"
#include "Symbols.h"
#include "lld/Common/BPSectionOrdererBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"

namespace lld::macho {

class InputSection;

class BPSymbolMacho : public BPSymbol {
  const Symbol *sym;

public:
  explicit BPSymbolMacho(const Symbol *s) : sym(s) {}

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

  const Symbol *getSymbol() const { return sym; }
};

class BPSectionMacho : public BPSectionBase {
  const InputSection *isec;
  uint64_t sectionIdx;
  mutable std::vector<std::unique_ptr<BPSymbol>> symbols;

public:
  explicit BPSectionMacho(const InputSection *sec, uint64_t sectionIdx)
      : isec(sec), sectionIdx(sectionIdx) {}

  const InputSection *getSection() const { return isec; }

  llvm::StringRef getName() const override { return isec->getName(); }

  uint64_t getSize() const override { return isec->getSize(); }

  uint64_t getSectionIdx() const { return sectionIdx; }

  bool isCodeSection() const override { return macho::isCodeSection(isec); }

  bool hasValidData() const override {
    return isec && !isec->data.empty() && isec->data.data();
  }

  llvm::ArrayRef<uint8_t> getSectionData() const override { return isec->data; }

  llvm::ArrayRef<std::unique_ptr<BPSymbol>> getSymbols() const override {
    for (auto *d : isec->symbols) {
      symbols.emplace_back(std::make_unique<BPSymbolMacho>(d));
    }
    return symbols;
  }

  // Linkage names can be prefixed with "_" or "l_" on Mach-O. See
  // Mangler::getNameWithPrefix() for details.
  bool needResolveLinkageName(llvm::StringRef &name) const override {
    return (name.consume_front("_") || name.consume_front("l_"));
  }

  void getSectionHash(llvm::SmallVectorImpl<uint64_t> &hashes) const override {
    constexpr unsigned windowSize = 4;

    // Calculate content hashes
    size_t dataSize = isec->data.size();
    for (size_t i = 0; i < dataSize; i++) {
      auto window = isec->data.drop_front(i).take_front(windowSize);
      hashes.push_back(xxHash64(window));
    }

    // Calculate relocation hashes
    for (const auto &r : isec->relocs) {
      if (r.length == 0 || r.referent.isNull() || r.offset >= isec->data.size())
        continue;

      uint64_t relocHash = getRelocHash(r, this);
      uint32_t start = (r.offset < windowSize) ? 0 : r.offset - windowSize + 1;
      for (uint32_t i = start; i < r.offset + r.length; i++) {
        auto window = isec->data.drop_front(i).take_front(windowSize);
        hashes.push_back(xxHash64(window) + relocHash);
      }
    }

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  const InputSection *getInputSection() const { return isec; }

  static bool classof(const BPSectionBase *s) { return true; }

private:
  static uint64_t getRelocHash(const Reloc &reloc,
                               const BPSectionMacho *section) {
    auto *isec = reloc.getReferentInputSection();
    std::optional<uint64_t> sectionIdx;
    if (isec && isec == section->getSection())
      sectionIdx = section->getSectionIdx();

    std::string kind;
    if (isec)
      kind = ("Section " + Twine(isec->kind())).str();

    if (auto *sym = reloc.referent.dyn_cast<Symbol *>()) {
      kind += (" Symbol " + Twine(sym->kind())).str();
      if (auto *d = llvm::dyn_cast<Defined>(sym)) {
        return BPSectionBase::getRelocHash(kind, sectionIdx.value_or(0),
                                           d->value, reloc.addend);
      }
    }
    return BPSectionBase::getRelocHash(kind, sectionIdx.value_or(0), 0,
                                       reloc.addend);
  }
};

/// Run Balanced Partitioning to find the optimal function and data order to
/// improve startup time and compressed size.
///
/// It is important that .subsections_via_symbols is used to ensure functions
/// and data are in their own sections and thus can be reordered.
llvm::DenseMap<const lld::macho::InputSection *, size_t>
runBalancedPartitioning(size_t &highestAvailablePriority,
                        llvm::StringRef profilePath,
                        bool forFunctionCompression, bool forDataCompression,
                        bool compressionSortStartupFunctions, bool verbose);

} // namespace lld::macho

#endif
