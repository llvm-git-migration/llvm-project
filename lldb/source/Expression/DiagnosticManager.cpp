//===-- DiagnosticManager.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DiagnosticManager.h"

#include "llvm/Support/ErrorHandling.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;
char DetailedExpressionError::ID;

static const char *StringForSeverity(lldb::Severity severity) {
  switch (severity) {
  // this should be exhaustive
  case lldb::eSeverityError:
    return "error: ";
  case lldb::eSeverityWarning:
    return "warning: ";
  case lldb::eSeverityInfo:
    return "";
  }
  llvm_unreachable("switch needs another case for lldb::Severity enum");
}

std::string DetailedExpressionError::message() const {
  std::string str;
  llvm::raw_string_ostream(str)
      << StringForSeverity(m_detail.severity) << m_detail.rendered;
  return str;
}

std::string DiagnosticManager::GetString(char separator) {
  std::string str;
  llvm::raw_string_ostream stream(str);

  for (const auto &diagnostic : Diagnostics()) {
    llvm::StringRef severity = StringForSeverity(diagnostic->GetSeverity());
    stream << severity;

    llvm::StringRef message = diagnostic->GetMessage();
    std::string searchable_message = message.lower();
    auto severity_pos = message.find(severity);
    stream << message.take_front(severity_pos);

    if (severity_pos != llvm::StringRef::npos)
      stream << message.drop_front(severity_pos + severity.size());
    stream << separator;
  }
  return str;
}

void DiagnosticManager::Dump(Log *log) {
  if (!log)
    return;

  std::string str = GetString();

  // We want to remove the last '\n' because log->PutCString will add
  // one for us.

  if (str.size() && str.back() == '\n')
    str.pop_back();

  log->PutString(str);
}

llvm::Error Diagnostic::GetAsError() const {
  return llvm::make_error<DetailedExpressionError>(m_detail);
}

llvm::Error
DiagnosticManager::GetAsError(lldb::ExpressionResults result) const {
  llvm::Error diags = Status::FromExpressionError(result, "").takeError();
  for (const auto &diagnostic : m_diagnostics)
    diags = llvm::joinErrors(std::move(diags), diagnostic->GetAsError());
  return diags;
}

llvm::Error
DiagnosticManager::GetAsError(llvm::Twine msg) const {
  llvm::Error diags = llvm::createStringError(msg);
  for (const auto &diagnostic : m_diagnostics)
    diags = llvm::joinErrors(std::move(diags), diagnostic->GetAsError());
  return diags;
}

void DiagnosticManager::AddDiagnostic(llvm::StringRef message,
                                      lldb::Severity severity,
                                      DiagnosticOrigin origin,
                                      uint32_t compiler_id) {
  m_diagnostics.emplace_back(std::make_unique<Diagnostic>(
      origin, compiler_id,
      DiagnosticDetail{{}, severity, message.str(), message.str()}));
}

size_t DiagnosticManager::Printf(lldb::Severity severity, const char *format,
                                 ...) {
  StreamString ss;

  va_list args;
  va_start(args, format);
  size_t result = ss.PrintfVarArg(format, args);
  va_end(args);

  AddDiagnostic(ss.GetString(), severity, eDiagnosticOriginLLDB);

  return result;
}

void DiagnosticManager::PutString(lldb::Severity severity,
                                  llvm::StringRef str) {
  if (str.empty())
    return;
  AddDiagnostic(str, severity, eDiagnosticOriginLLDB);
}

void Diagnostic::AppendMessage(llvm::StringRef message,
                               bool precede_with_newline) {
  if (precede_with_newline) {
    m_detail.message.push_back('\n');
    m_detail.rendered.push_back('\n');
  }
  m_detail.message += message;
  m_detail.rendered += message;
}
