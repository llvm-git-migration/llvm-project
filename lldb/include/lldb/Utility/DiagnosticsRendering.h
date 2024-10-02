//===-- DiagnosticsRendering.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_DIAGNOSTICSRENDERING_H
#define LLDB_UTILITY_DIAGNOSTICSRENDERING_H

#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "llvm/Support/WithColor.h"

namespace lldb_private {

llvm::raw_ostream &PrintSeverity(Stream &stream, lldb::Severity severity);

void RenderDiagnosticDetails(Stream &stream,
                             std::optional<uint16_t> offset_in_command,
                             bool show_inline,
                             llvm::ArrayRef<DiagnosticDetail> details);
} // namespace lldb_private
#endif
