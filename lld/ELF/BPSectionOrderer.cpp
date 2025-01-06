//===- BPSectionOrderer.cpp------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "lld/Common/BPSectionOrdererBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/xxhash.h"

#include "SymbolTable.h"
#include "Symbols.h"

using namespace llvm;
using namespace lld::elf;

void BPSectionELF::getSectionHashes(
    llvm::SmallVectorImpl<uint64_t> &hashes,
    const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) const {
  constexpr unsigned windowSize = 4;

  size_t size = isec->content().size();
  for (size_t i = 0; i != size; ++i) {
    auto window = isec->content().drop_front(i).take_front(windowSize);
    hashes.push_back(xxHash64(window));
  }

  llvm::sort(hashes);
  hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
}

llvm::DenseMap<const lld::elf::InputSectionBase *, int>
lld::elf::runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                                  bool forFunctionCompression,
                                  bool forDataCompression,
                                  bool compressionSortStartupFunctions,
                                  bool verbose) {
  size_t highestAvailablePriority = std::numeric_limits<int>::max();
  // Collect all InputSectionBase objects from symbols and wrap them as
  // BPSectionELF instances for balanced partitioning which follow the way
  // '--symbol-ordering-file' does.
  SmallVector<std::unique_ptr<BPSectionBase>> sections;

  for (Symbol *sym : ctx.symtab->getSymbols())
    if (sym->getSize() > 0)
      if (auto *d = dyn_cast<Defined>(sym))
        if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section))
          sections.emplace_back(std::make_unique<BPSectionELF>(
              sec, std::make_unique<BPSymbolELF>(d)));

  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols())
      if (sym->getSize() > 0)
        if (auto *d = dyn_cast<Defined>(sym))
          if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section))
            sections.emplace_back(std::make_unique<BPSectionELF>(
                sec, std::make_unique<BPSymbolELF>(d)));

  auto reorderedSections = BPSectionBase::reorderSectionsByBalancedPartitioning(
      highestAvailablePriority, profilePath, forFunctionCompression,
      forDataCompression, compressionSortStartupFunctions, verbose, sections);

  DenseMap<const InputSectionBase *, int> result;
  for (const auto [sec, priority] : reorderedSections) {
    auto *elfSection = cast<BPSectionELF>(sec);
    result.try_emplace(
        static_cast<const InputSectionBase *>(elfSection->getSection()),
        static_cast<int>(priority));
  }
  return result;
}
