//===------------------------------ Client.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

// Returns where to store log files and socket address. Of the format
// /tmp/clang-<BLAKE3HashOfClagnFullVersion>/
std::string getBasePath();

llvm::Error attemptHandshake(int SocketFD, DiagnosticsEngine &Diag);

llvm::Error spawnModuleBuildDaemon(StringRef BasePath, const char *Argv0,
                                   DiagnosticsEngine &Diag);

Expected<int> getModuleBuildDaemon(const char *Argv0, StringRef BasePath,
                                   DiagnosticsEngine &Diag);

void spawnModuleBuildDaemonAndHandshake(const CompilerInvocation &Clang,
                                        DiagnosticsEngine &Diag,
                                        const char *Argv0);

} // namespace cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H
