//===-- DiagnosticManager.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_DIAGNOSTICMANAGER_H
#define LLDB_EXPRESSION_DIAGNOSTICMANAGER_H

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace lldb_private {

/// A compiler-independent representation of a Diagnostic. Expression
/// evaluation failures often have more than one diagnostic that a UI
/// layer might want to render differently, for example to colorize
/// it.
///
/// Running example:
///   (lldb) expr 1+foo
///   error: <user expression 0>:1:3: use of undeclared identifier 'foo'
///   1+foo
///     ^
struct DiagnosticDetail {
  struct SourceLocation {
    FileSpec file;
    unsigned line = 0;
    uint16_t column = 0;
    uint16_t length = 0;
    bool in_user_input = false;
  };
  /// Contains {{}, 1, 3, 3, true} in the example above.
  std::optional<SourceLocation> source_location;
  /// Contains eSeverityError in the example above.
  lldb::Severity severity = lldb::eSeverityInfo;
  /// Contains "use of undeclared identifier 'x'" in the example above.
  std::string message;
  /// Contains the fully rendered error message.
  std::string rendered;
};

/// An llvm::Error used to communicate diagnostics in Status. Multiple
/// diagnostics may be chained in an llvm::ErrorList.
class DetailedExpressionError
    : public llvm::ErrorInfo<DetailedExpressionError, llvm::ECError> {
  DiagnosticDetail m_detail;

public:
  using llvm::ErrorInfo<DetailedExpressionError, llvm::ECError>::ErrorInfo;
  DetailedExpressionError(DiagnosticDetail detail) {}
  std::string message() const override;
  static char ID;
};

enum DiagnosticOrigin {
  eDiagnosticOriginUnknown = 0,
  eDiagnosticOriginLLDB,
  eDiagnosticOriginClang,
  eDiagnosticOriginSwift,
  eDiagnosticOriginLLVM
};

const uint32_t LLDB_INVALID_COMPILER_ID = UINT32_MAX;

class Diagnostic {
  friend class DiagnosticManager;

public:
  DiagnosticOrigin getKind() const { return m_origin; }

  static bool classof(const Diagnostic *diag) {
    DiagnosticOrigin kind = diag->getKind();
    switch (kind) {
    case eDiagnosticOriginUnknown:
    case eDiagnosticOriginLLDB:
    case eDiagnosticOriginLLVM:
      return true;
    case eDiagnosticOriginClang:
    case eDiagnosticOriginSwift:
      return false;
    }
  }

  Diagnostic(DiagnosticOrigin origin, uint32_t compiler_id,
             DiagnosticDetail detail)
      : m_origin(origin), m_compiler_id(compiler_id), m_detail(detail) {}

  virtual ~Diagnostic() = default;

  virtual bool HasFixIts() const { return false; }

  lldb::Severity GetSeverity() const { return m_detail.severity; }

  uint32_t GetCompilerID() const { return m_compiler_id; }

  llvm::StringRef GetMessage() const { return m_detail.message; }
  llvm::Error GetAsError() const;

  void AppendMessage(llvm::StringRef message, bool precede_with_newline = true);

protected:
  DiagnosticOrigin m_origin;
  /// Compiler-specific diagnostic ID.
  uint32_t m_compiler_id;
  DiagnosticDetail m_detail;
};

typedef std::vector<std::unique_ptr<Diagnostic>> DiagnosticList;

class DiagnosticManager {
public:
  void Clear() {
    m_diagnostics.clear();
    m_fixed_expression.clear();
  }

  const DiagnosticList &Diagnostics() { return m_diagnostics; }

  bool HasFixIts() const {
    return llvm::any_of(m_diagnostics,
                        [](const std::unique_ptr<Diagnostic> &diag) {
                          return diag->HasFixIts();
                        });
  }

  void AddDiagnostic(llvm::StringRef message, lldb::Severity severity,
                     DiagnosticOrigin origin,
                     uint32_t compiler_id = LLDB_INVALID_COMPILER_ID);

  void AddDiagnostic(std::unique_ptr<Diagnostic> diagnostic) {
    if (diagnostic)
      m_diagnostics.push_back(std::move(diagnostic));
  }

  /// Moves over the contents of a second diagnostic manager over. Leaves other
  /// diagnostic manager in an empty state.
  void Consume(DiagnosticManager &&other) {
    std::move(other.m_diagnostics.begin(), other.m_diagnostics.end(),
              std::back_inserter(m_diagnostics));
    m_fixed_expression = std::move(other.m_fixed_expression);
    other.Clear();
  }

  size_t Printf(lldb::Severity severity, const char *format, ...)
      __attribute__((format(printf, 3, 4)));
  void PutString(lldb::Severity severity, llvm::StringRef str);

  void AppendMessageToDiagnostic(llvm::StringRef str) {
    if (!m_diagnostics.empty())
      m_diagnostics.back()->AppendMessage(str);
  }

  /// Copies the diagnostics into an llvm::Error{List}.
  /// The first diagnostic wraps \c result.
  llvm::Error GetAsError(lldb::ExpressionResults result) const;

  /// Copies the diagnostics into an llvm::Error, the first diagnostic
  /// being an llvm::StringError.
  llvm::Error GetAsError(llvm::Twine msg) const;

  void Dump(Log *log);

  const std::string &GetFixedExpression() { return m_fixed_expression; }

  // Moves fixed_expression to the internal storage.
  void SetFixedExpression(std::string fixed_expression) {
    m_fixed_expression = std::move(fixed_expression);
  }

protected:
  DiagnosticList m_diagnostics;
  std::string m_fixed_expression;
};
}

#endif // LLDB_EXPRESSION_DIAGNOSTICMANAGER_H
