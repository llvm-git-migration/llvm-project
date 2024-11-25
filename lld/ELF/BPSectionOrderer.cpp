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
#include "lld/Common/CommonLinkerContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Support/TimeProfiler.h"

#include "SymbolTable.h"
#include "Symbols.h"

using namespace llvm;
using namespace lld::elf;

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
              sec, std::make_unique<BPSymbolELF>(sym)));

  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols())
      if (sym->getSize() > 0)
        if (auto *d = dyn_cast<Defined>(sym))
          if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section))
            sections.emplace_back(std::make_unique<BPSectionELF>(
                sec, std::make_unique<BPSymbolELF>(sym)));

  auto reorderedSections =
      lld::BPSectionOrdererBase::reorderSectionsByBalancedPartitioning(
          highestAvailablePriority, profilePath, forFunctionCompression,
          forDataCompression, compressionSortStartupFunctions, verbose,
          sections);

  DenseMap<const InputSectionBase *, int> result;
  for (const auto &[sec, priority] : reorderedSections) {
    auto *elfSection = cast<BPSectionELF>(sec);
    result.try_emplace(elfSection->getSymbol()->getInputSection(),
                       static_cast<int>(priority));
  }
  return result;
}
