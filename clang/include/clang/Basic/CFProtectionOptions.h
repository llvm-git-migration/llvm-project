//===--- CFProtectionOptions.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines constants for -fcf-protection and other related flags.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H
#define LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H

namespace clang {

enum class CFBranchLabelSchemeKind { Default, Unlabeled, FuncSig };

} // namespace clang

#endif // #ifndef LLVM_CLANG_BASIC_CFPROTECTIONOPTIONS_H
