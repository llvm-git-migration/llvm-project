//===-- ClangInstallAPI.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to clang-installapi; it is a wrapper
// for functionality in the InstallAPI clang library.
//
//===----------------------------------------------------------------------===//

#include "Options.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/InstallAPI/Frontend.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TextAPI/RecordVisitor.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include <memory>

using namespace clang;
using namespace clang::installapi;
using namespace clang::driver::options;
using namespace llvm::opt;
using namespace llvm::MachO;

static const ArgStringList *
getCC1Arguments(clang::DiagnosticsEngine &Diags,
                clang::driver::Compilation *Compilation) {
  const auto &Jobs = Compilation->getJobs();
  if (Jobs.size() != 1 || !isa<clang::driver::Command>(*Jobs.begin())) {
    SmallString<256> error_msg;
    llvm::raw_svector_ostream error_stream(error_msg);
    Jobs.Print(error_stream, "; ", true);
    Diags.Report(clang::diag::err_fe_expected_compiler_job)
        << error_stream.str();
    return nullptr;
  }

  // The one job we find should be to invoke clang again.
  const auto &Cmd = cast<clang::driver::Command>(*Jobs.begin());
  if (StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags.Report(clang::diag::err_fe_expected_clang_command);
    return nullptr;
  }

  return &Cmd.getArguments();
}

static CompilerInvocation *createInvocation(clang::DiagnosticsEngine &Diags,
                                            const ArgStringList &cc1Args) {
  assert(!cc1Args.empty() && "Must at least contain the program name!");
  CompilerInvocation *Invocation = new CompilerInvocation;
  CompilerInvocation::CreateFromArgs(*Invocation, cc1Args, Diags);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  return Invocation;
}

static bool runFrontend(StringRef ProgName, bool Verbose,
                        const InstallAPIContext &Ctx,
                        clang::driver::Driver &Driver, CompilerInstance &CI,
                        const ArrayRef<std::string> InitialArgs) {

  std::unique_ptr<llvm::MemoryBuffer> ProcessedInput = createInputBuffer(Ctx);
  // Skip invoking cc1 when there are no header inputs.
  if (!ProcessedInput)
    return true;

  if (Verbose)
    llvm::errs() << getName(Ctx.Type) << " Headers:\n"
                 << ProcessedInput->getBuffer() << "\n";

  // Reconstruct arguments with unique values like target triple or input
  // headers.
  std::vector<const char *> Args = {ProgName.data(), "-target",
                                    Ctx.Records->getTriple().str().c_str()};
  llvm::transform(InitialArgs, std::back_inserter(Args),
                  [](const std::string &A) { return A.c_str(); });
  Args.push_back(ProcessedInput->getBufferIdentifier().data());

  // Set up compilation, invocation, and action to execute.
  const std::unique_ptr<clang::driver::Compilation> Compilation(
      Driver.BuildCompilation(Args));
  if (!Compilation)
    return false;
  const llvm::opt::ArgStringList *const CC1Args =
      getCC1Arguments(*Ctx.Diags, Compilation.get());
  if (!CC1Args)
    return false;
  std::unique_ptr<clang::CompilerInvocation> Invocation(
      createInvocation(*Ctx.Diags, *CC1Args));

  if (Verbose) {
    llvm::errs() << "CC1 Invocation:\n";
    Compilation->getJobs().Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  Invocation->getPreprocessorOpts().addRemappedFile(
      ProcessedInput->getBufferIdentifier(), ProcessedInput.release());

  CI.setInvocation(std::move(Invocation));
  CI.setFileManager(Ctx.FM);
  auto Action = std::make_unique<InstallAPIAction>(*Ctx.Records);
  CI.createDiagnostics();
  if (!CI.hasDiagnostics())
    return false;
  CI.createSourceManager(*Ctx.FM);
  return CI.ExecuteAction(*Action);
}

static bool run(ArrayRef<const char *> Args, const char *ProgName) {
  // Setup Diagnostics engine.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  const llvm::opt::OptTable &ClangOpts = clang::driver::getDriverOptTable();
  unsigned MissingArgIndex, MissingArgCount;
  llvm::opt::InputArgList ParsedArgs = ClangOpts.ParseArgs(
      ArrayRef(Args).slice(1), MissingArgIndex, MissingArgCount);
  ParseDiagnosticArgs(*DiagOpts, ParsedArgs);

  IntrusiveRefCntPtr<DiagnosticsEngine> Diag = new clang::DiagnosticsEngine(
      new clang::DiagnosticIDs(), DiagOpts.get(),
      new clang::TextDiagnosticPrinter(llvm::errs(), DiagOpts.get()));

  // Create file manager for all file operations.
  IntrusiveRefCntPtr<clang::FileManager> FM(
      new FileManager(clang::FileSystemOptions()));

  // Set up driver to parse input arguments.
  auto DriverArgs = llvm::ArrayRef(Args).slice(1);
  clang::driver::Driver Driver(ProgName, llvm::sys::getDefaultTargetTriple(),
                               *Diag, "clang installapi tool");
  Driver.setInstalledDir(llvm::sys::path::parent_path(ProgName));
  auto TargetAndMode =
      clang::driver::ToolChain::getTargetAndModeFromProgramName(ProgName);
  Driver.setTargetAndMode(TargetAndMode);
  bool HasError = false;
  llvm::opt::InputArgList ArgList =
      Driver.ParseArgStrings(DriverArgs, /*UseDriverMode=*/true, HasError);
  if (HasError)
    return EXIT_FAILURE;
  Driver.setCheckInputsExist(false);

  // Capture InstallAPI specific options and diagnose any option errors.
  Options Opts(*Diag, FM.get(), ArgList);
  if (Diag->hasErrorOccurred())
    return EXIT_FAILURE;

  InstallAPIContext Ctx = Opts.createContext();
  if (Diag->hasErrorOccurred())
    return EXIT_FAILURE;

  // Set up compilation.
  std::unique_ptr<CompilerInstance> CI(new CompilerInstance());
  CI->setFileManager(FM.get());
  CI->createDiagnostics();
  if (!CI->hasDiagnostics())
    return EXIT_FAILURE;

  // Execute and gather AST results.
  llvm::MachO::Records FrontendResults;
  for (const auto &[Targ, Trip] : Opts.DriverOpts.Targets) {
    for (const HeaderType Type :
         {HeaderType::Public, HeaderType::Private, HeaderType::Project}) {
      Ctx.Records = std::make_shared<RecordsSlice>(Trip);
      Ctx.Type = Type;
      if (!runFrontend(ProgName, Opts.DriverOpts.Verbose, Ctx, Driver, *CI,
                       Opts.getClangFrontendArgs()))
        return EXIT_FAILURE;
      FrontendResults.emplace_back(std::move(Ctx.Records));
    }
  }

  // After symbols have been collected, prepare to write output.
  auto Out = CI->createOutputFile(Ctx.OutputLoc, /*Binary=*/false,
                                  /*RemoveFileOnSignal=*/false,
                                  /*UseTemporary=*/false,
                                  /*CreateMissingDirectories=*/false);
  if (!Out)
    return EXIT_FAILURE;

  // Assign attributes for serialization.
  auto Symbols = std::make_unique<SymbolSet>();
  for (const auto &FR : FrontendResults) {
    SymbolConverter Converter(Symbols.get(), FR->getTarget());
    FR->visit(Converter);
  }

  InterfaceFile IF(std::move(Symbols));
  for (const auto &TargetInfo : Opts.DriverOpts.Targets) {
    IF.addTarget(TargetInfo.first);
    IF.setFromBinaryAttrs(Ctx.BA, TargetInfo.first);
  }

  // Write output file and perform CI cleanup.
  if (auto Err = TextAPIWriter::writeToStream(*Out, IF, Ctx.FT)) {
    Diag->Report(diag::err_cannot_open_file) << Ctx.OutputLoc;
    CI->clearOutputFiles(/*EraseFiles=*/true);
    return EXIT_FAILURE;
  }

  CI->clearOutputFiles(/*EraseFiles=*/false);
  return EXIT_SUCCESS;
}

int clang_installapi_main(int argc, char **argv,
                          const llvm::ToolContext &ToolContext) {
  // Standard set up, so program fails gracefully.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram StackPrinter(argc, argv);
  llvm::llvm_shutdown_obj Shutdown;

  if (llvm::sys::Process::FixupStandardFileDescriptors())
    return EXIT_FAILURE;

  const char *ProgName =
      ToolContext.NeedsPrependArg ? ToolContext.PrependArg : ToolContext.Path;
  return run(llvm::ArrayRef(argv, argc), ProgName);
}
