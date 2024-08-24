//===--- Interpreter.h - Incremental Compilation and Execution---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the component which performs incremental code
// compilation and execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_INTERPRETER_H
#define LLVM_CLANG_INTERPRETER_INTERPRETER_H

#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Ownership.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace llvm {
namespace orc {
class LLJIT;
class LLJITBuilder;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {

class CompilerInstance;

class IncrementalExecutor;
class IncrementalParser;

/// Create a pre-configured \c CompilerInstance for incremental processing.
class IncrementalCompilerBuilder {
public:
  IncrementalCompilerBuilder() {}

  void SetCompilerArgs(const std::vector<const char *> &Args) {
    UserArgs = Args;
  }

  void SetTargetTriple(std::string TT) { TargetTriple = TT; }

  // General C++
  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCpp();

  // Offload options
  void SetOffloadArch(llvm::StringRef Arch) { OffloadArch = Arch; };

  // CUDA specific
  void SetCudaSDK(llvm::StringRef path) { CudaSDKPath = path; };

  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCudaHost();
  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCudaDevice();

private:
  static llvm::Expected<std::unique_ptr<CompilerInstance>>
  create(std::string TT, std::vector<const char *> &ClangArgv);

  llvm::Expected<std::unique_ptr<CompilerInstance>> createCuda(bool device);

  std::vector<const char *> UserArgs;
  std::optional<std::string> TargetTriple;

  llvm::StringRef OffloadArch;
  llvm::StringRef CudaSDKPath;
};

class Interpreter;
/// Provides a callback class allowing to listen to interpreter events and to
/// specialize some operations.
class InterpreterCallbacks {
  Interpreter &Interp;

public:
  InterpreterCallbacks(Interpreter &I) : Interp(I) {}
  virtual ~InterpreterCallbacks();
  virtual void ProcessingTopLevelStmtDecl(TopLevelStmtDecl *D);
};

/// Provides top-level interfaces for incremental compilation and execution.
class Interpreter {
  friend Value;

  std::unique_ptr<llvm::orc::ThreadSafeContext> TSCtx;
  std::unique_ptr<InterpreterCallbacks> InterpreterCB;
  std::unique_ptr<IncrementalParser> IncrParser;
  std::unique_ptr<IncrementalExecutor> IncrExecutor;

  // An optional parser for CUDA offloading
  std::unique_ptr<IncrementalParser> DeviceParser;

  unsigned InitPTUSize = 0;

  // This member holds the last result of the value printing. It's a class
  // member because we might want to access it after more inputs. If no value
  // printing happens, it's in an invalid state.
  Value LastValue;

  // The cached declaration of std::string used as a return type for the built
  // trampoline. This is done in C++ to simplify the memory management for
  // user-defined printing functions.
  Decl *StdString = nullptr;

  // A cache for the compiled destructors used to for de-allocation of managed
  // clang::Values.
  llvm::DenseMap<CXXRecordDecl *, llvm::orc::ExecutorAddr> Dtors;

  std::array<Expr *, 4> ValuePrintingInfo = {0};

protected:
  // Derived classes can use an extended interface of the Interpreter.
  Interpreter(std::unique_ptr<CompilerInstance> CI, llvm::Error &Err,
              std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder = nullptr);

  // Create the internal IncrementalExecutor, or re-create it after calling
  // ResetExecutor().
  llvm::Error CreateExecutor();

  // Delete the internal IncrementalExecutor. This causes a hard shutdown of the
  // JIT engine. In particular, it doesn't run cleanup or destructors.
  void ResetExecutor();

public:
  virtual ~Interpreter();
  static llvm::Expected<std::unique_ptr<Interpreter>>
  create(std::unique_ptr<CompilerInstance> CI);
  static llvm::Expected<std::unique_ptr<Interpreter>>
  createWithCUDA(std::unique_ptr<CompilerInstance> CI,
                 std::unique_ptr<CompilerInstance> DCI);
  const ASTContext &getASTContext() const;
  ASTContext &getASTContext();
  const CompilerInstance *getCompilerInstance() const;
  CompilerInstance *getCompilerInstance();
  llvm::Expected<llvm::orc::LLJIT &> getExecutionEngine();

  llvm::Expected<PartialTranslationUnit &> Parse(llvm::StringRef Code);
  llvm::Error Execute(PartialTranslationUnit &T);
  llvm::Error ParseAndExecute(llvm::StringRef Code, Value *V = nullptr);

  /// Undo N previous incremental inputs.
  llvm::Error Undo(unsigned N = 1);

  /// Link a dynamic library
  llvm::Error LoadDynamicLibrary(const char *name);

  /// \returns the \c ExecutorAddr of a \c GlobalDecl. This interface uses
  /// the CodeGenModule's internal mangling cache to avoid recomputing the
  /// mangled name.
  llvm::Expected<llvm::orc::ExecutorAddr> getSymbolAddress(GlobalDecl GD) const;

  /// \returns the \c ExecutorAddr of a given name as written in the IR.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef IRName) const;

  /// \returns the \c ExecutorAddr of a given name as written in the object
  /// file.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddressFromLinkerName(llvm::StringRef LinkerName) const;

  InterpreterCallbacks *getInterpreterCallbacks() {
    return InterpreterCB.get();
  }
  const InterpreterCallbacks *getInterpreterCallbacks() const {
    return const_cast<Interpreter *>(this)->getInterpreterCallbacks();
  }
  void setInterpreterCallbacks(std::unique_ptr<InterpreterCallbacks> CB) {
    InterpreterCB = std::move(CB);
  }

  llvm::Expected<Expr *> SynthesizeExpr(Expr *E);

  std::unique_ptr<llvm::Module> GenModule();

private:
  size_t getEffectivePTUSize() const;
  void markUserCodeStart();

  std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder;

  std::string ValueDataToString(const Value &V);
  std::string ValueTypeToString(const Value &V) const;

  // When we deallocate clang::Value we need to run the destructor of the type.
  // This function forces emission of the needed dtor.
  llvm::Expected<llvm::orc::ExecutorAddr> CompileDtorCall(CXXRecordDecl *CXXRD);
};
} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_INTERPRETER_H
