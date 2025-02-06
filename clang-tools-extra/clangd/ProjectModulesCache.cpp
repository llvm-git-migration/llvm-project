//===------------------ ProjectModulesCache.cpp ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectModulesCache.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>

namespace clang::clangd {
namespace {
class SharedProjectModulesCache : public ProjectModulesCache {
public:
  std::optional<std::string>
  getSourceForModuleName(llvm::StringRef ModuleName,
                         PathRef RequiredSrcFile = PathRef()) override {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto Iter = ModuleNameToSource.find(ModuleName);
    if (Iter == ModuleNameToSource.end())
      return std::nullopt;

    return Iter->second;
  }

  void clearEntry(llvm::StringRef ModuleName,
                  PathRef RequiredSrcFile = PathRef()) override {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto Iter = ModuleNameToSource.find(ModuleName);
    if (Iter == ModuleNameToSource.end())
      return;

    ModuleNameToSource.erase(Iter);
  }

  void setEntry(PathRef FilePath, llvm::StringRef ModuleName) override {
    std::lock_guard<std::mutex> Lock(Mutex);

    ModuleNameToSource[ModuleName] = FilePath;
  }

private:
  std::mutex Mutex;
  llvm::StringMap<std::string> ModuleNameToSource;
};
} // namespace

std::unique_ptr<ProjectModulesCache> createProjectModulesCache() {
  return std::make_unique<SharedProjectModulesCache>();
}

} // namespace clang::clangd
