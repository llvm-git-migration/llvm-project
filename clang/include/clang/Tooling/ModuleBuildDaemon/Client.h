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
#include "llvm/Support/raw_socket_stream.h"

namespace clang::tooling::cc1modbuildd {

llvm::Error attemptHandshake(llvm::raw_socket_stream &Client,
                             DiagnosticsEngine &Diag);

llvm::Error spawnModuleBuildDaemon(llvm::StringRef BasePath, const char *Argv0,
                                   clang::DiagnosticsEngine &Diag);

llvm::Expected<std::unique_ptr<llvm::raw_socket_stream>>
getModuleBuildDaemon(const char *Argv0, llvm::StringRef BasePath,
                     clang::DiagnosticsEngine &Diag);

void spawnModuleBuildDaemonAndHandshake(const clang::CompilerInvocation &Clang,
                                        clang::DiagnosticsEngine &Diag,
                                        const char *Argv0);

} // namespace clang::tooling::cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_CLIENT_H
