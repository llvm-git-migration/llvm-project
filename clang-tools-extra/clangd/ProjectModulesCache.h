//===------------------ ProjectModulesCache.h --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROJECTMODULESCACHE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROJECTMODULESCACHE_H

#include "support/Path.h"
#include <memory>

namespace clang {
namespace clangd {

/// A cache for the module name to file name map in the project.
/// The rationale is:
///   (1) It is fast to get the module name to a file.
///   (2) It may be slow to get a file with a specified module name.
///   (3) The module name of files may not change drastically and frequently.
///
/// The cache itself is not responsible for the validness of cached result.
/// Users of the cache should check it after getting the result and updating
/// the cache if the result is invalid.
class ProjectModulesCache {
public:
  virtual ~ProjectModulesCache() = default;

  virtual std::optional<std::string>
  getSourceForModuleName(llvm::StringRef ModuleName,
                         PathRef RequiredSrcFile = PathRef()) = 0;

  virtual void clearEntry(llvm::StringRef ModuleName,
                          PathRef RequiredSrcFile = PathRef()) = 0;

  virtual void setEntry(PathRef FilePath, llvm::StringRef ModuleName) = 0;
};

std::unique_ptr<ProjectModulesCache> createProjectModulesCache();

} // namespace clangd
} // namespace clang

#endif
