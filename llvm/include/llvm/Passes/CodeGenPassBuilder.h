//===- Construction of codegen pass pipelines ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for producing common pass manager configurations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_CODEGENPASSBUILDER_H
#define LLVM_PASSES_CODEGENPASSBUILDER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/AssignmentTrackingAnalysis.h"
#include "llvm/CodeGen/CallBrPrepare.h"
#include "llvm/CodeGen/CodeGenPrepare.h"
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
#include "llvm/CodeGen/DwarfEHPrepare.h"
#include "llvm/CodeGen/ExpandMemCmp.h"
#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/CodeGen/FreeMachineFunction.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/GlobalMerge.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/InterleavedLoadCombine.h"
#include "llvm/CodeGen/JMCInstrumenter.h"
#include "llvm/CodeGen/LowerEmuTLS.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/CodeGen/SafeStack.h"
#include "llvm/CodeGen/SelectOptimize.h"
#include "llvm/CodeGen/ShadowStackGCLowering.h"
#include "llvm/CodeGen/SjLjEHPrepare.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/CodeGen/WasmEHPrepare.h"
#include "llvm/CodeGen/WinEHPrepare.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/CFGuard.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include <cassert>
#include <stack>
#include <type_traits>
#include <utility>

namespace llvm {

// FIXME: Dummy target independent passes definitions that have not yet been
// ported to new pass manager. Once they do, remove these.
#define DUMMY_FUNCTION_PASS(NAME, PASS_NAME)                                   \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_MODULE_PASS(NAME, PASS_NAME)                             \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME)                           \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(MachineFunction &,                                   \
                          MachineFunctionAnalysisManager &) {                  \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#include "llvm/Passes/MachinePassRegistry.def"

/// This class provides access to building LLVM's passes.
///
/// Its members provide the baseline state available to passes during their
/// construction. The \c MachinePassRegistry.def file specifies how to construct
/// all of the built-in passes, and those may reference these members during
/// construction.
template <typename DerivedT, typename TargetMachineT> class CodeGenPassBuilder {
public:
  explicit CodeGenPassBuilder(TargetMachineT &TM,
                              const CGPassBuilderOption &Opts,
                              PassInstrumentationCallbacks *PIC)
      : TM(TM), Opt(Opts), PIC(PIC) {
    // Target should override TM.Options.EnableIPRA in their target-specific
    // LLVMTM ctor. See TargetMachine::setGlobalISel for example.
    if (Opt.EnableIPRA)
      TM.Options.EnableIPRA = *Opt.EnableIPRA;

    if (Opt.EnableGlobalISelAbort)
      TM.Options.GlobalISelAbort = *Opt.EnableGlobalISelAbort;

    if (!Opt.OptimizeRegAlloc)
      Opt.OptimizeRegAlloc = getOptLevel() != CodeGenOptLevel::None;
  }

  Error buildPipeline(ModulePassManager &MPM, raw_pwrite_stream &Out,
                      raw_pwrite_stream *DwoOut, CodeGenFileType FileType);

  PassInstrumentationCallbacks *getPassInstrumentationCallbacks() {
    return PIC;
  }

  /// Add one pass to pass manager, it can handle pass nesting automatically.
  template <typename PassT> void addPass(PassT &&Pass) {
    using ResultT = decltype(derived().template substitutePass<PassT>());
    constexpr bool IsVoid = std::is_void_v<ResultT>;
    StringRef PassName = std::conditional_t<IsVoid, PassT, ResultT>::name();

    if (!runBeforeAdding(PassName))
      return;

    if constexpr (IsVoid)
      addPassImpl(std::forward<PassT>(Pass));
    else
      addPassImpl(derived().template substitutePass<PassT>());

    runAfterAdding(PassName);
  }

  /// Add multiple passes.
  template <typename PassT, typename... PassTs>
  void addPass(PassT &&Pass, PassTs &&...Passes) {
    addPass(std::forward<PassT>(Pass));
    addPass(std::forward<PassTs>(Passes)...);
  }

  /// Add multiple passes with default constructor.
  template <typename... PassTs> void addPass() { addPass(PassTs()...); }

  /// @brief Substitute PassT with given pass, target can specialize this
  /// function template or override it in subclass, if it is overriden by
  /// subclass, it must return CodeGenPassBuilder::substitutePass() to get
  /// default return value. See also
  /// unittests/CodeGen/CodeGenPassBuilderTest.cpp.
  /// @tparam PassT The pass type that needs to be replaced.
  /// @return The replaced pass if substitution occurs, otherwise return void.
  template <typename PassT> auto substitutePass() {}

protected:
  template <typename PassT>
  using is_module_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Module &>(), std::declval<ModuleAnalysisManager &>()));

  template <typename PassT>
  static constexpr bool is_module_pass_v =
      is_detected<is_module_pass_t, PassT>::value;

  template <typename PassT>
  using is_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<Function &>(), std::declval<FunctionAnalysisManager &>()));

  template <typename PassT>
  static constexpr bool is_function_pass_v =
      is_detected<is_function_pass_t, PassT>::value;

  template <typename PassT>
  using is_machine_function_pass_t = decltype(std::declval<PassT &>().run(
      std::declval<MachineFunction &>(),
      std::declval<MachineFunctionAnalysisManager &>()));

  template <typename PassT>
  static constexpr bool is_machine_function_pass_v =
      is_detected<is_machine_function_pass_t, PassT>::value;

  TargetMachineT &TM;
  CGPassBuilderOption Opt;
  PassInstrumentationCallbacks *PIC;

  CodeGenOptLevel getOptLevel() { return TM.getOptLevel(); }

  /// Check whether or not GlobalISel should abort on error.
  /// When this is disabled, GlobalISel will fall back on SDISel instead of
  /// erroring out.
  bool isGlobalISelAbortEnabled() {
    return TM.Options.GlobalISelAbort == GlobalISelAbortMode::Enable;
  }

  /// Check whether or not a diagnostic should be emitted when GlobalISel
  /// uses the fallback path. In other words, it will emit a diagnostic
  /// when GlobalISel failed and isGlobalISelAbortEnabled is false.
  bool reportDiagnosticWhenGlobalISelFallback() {
    return TM.Options.GlobalISelAbort == GlobalISelAbortMode::DisableWithDiag;
  }

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  Error addInstSelector() {
    return make_error<StringError>("addInstSelector is not overridden",
                                   inconvertibleErrorCode());
  }

  /// Target can override this to add GlobalMergePass before all IR passes.
  void addGlobalMergePass() {}

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  void addILPOpts() {}

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  void addPreRegAlloc() {}

  /// addPreRewrite - Add passes to the optimized register allocation pipeline
  /// after register allocation is complete, but before virtual registers are
  /// rewritten to physical registers.
  ///
  /// These passes must preserve VirtRegMap and LiveIntervals, and when running
  /// after RABasic or RAGreedy, they should take advantage of LiveRegMatrix.
  /// When these passes run, VirtRegMap contains legal physreg assignments for
  /// all virtual registers.
  ///
  /// Note if the target overloads addRegAssignAndRewriteOptimized, this may not
  /// be honored. This is also not generally used for the fast variant,
  /// where the allocation and rewriting are done in one pass.
  void addPreRewrite() {}

  /// Add passes to be run immediately after virtual registers are rewritten
  /// to physical registers.
  void addPostRewrite() {}

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  void addPostRegAlloc() {}

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  void addPreSched2() {}

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  void addPreEmitPass() {}

  /// Targets may add passes immediately before machine code is emitted in this
  /// callback. This is called even later than `addPreEmitPass`.
  // FIXME: Rename `addPreEmitPass` to something more sensible given its actual
  // position and remove the `2` suffix here as this callback is what
  // `addPreEmitPass` *should* be but in reality isn't.
  void addPreEmitPass2() {}

  /// {{@ For GlobalISel
  ///

  /// addPreISel - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  void addPreISel() { llvm_unreachable("addPreISel is not overridden"); }

  /// This method should install an IR translator pass, which converts from
  /// LLVM code to machine instructions with possibly generic opcodes.
  Error addIRTranslator() {
    return make_error<StringError>("addIRTranslator is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before legalization.
  void addPreLegalizeMachineIR() {}

  /// This method should install a legalize pass, which converts the instruction
  /// sequence into one that can be selected by the target.
  Error addLegalizeMachineIR() {
    return make_error<StringError>("addLegalizeMachineIR is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the register bank selection.
  void addPreRegBankSelect() {}

  /// This method should install a register bank selector pass, which
  /// assigns register banks to virtual registers without a register
  /// class or register banks.
  Error addRegBankSelect() {
    return make_error<StringError>("addRegBankSelect is not overridden",
                                   inconvertibleErrorCode());
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before the (global) instruction selection.
  void addPreGlobalInstructionSelect() {}

  /// This method should install a (global) instruction selector pass, which
  /// converts possibly generic instructions to fully target-specific
  /// instructions, thereby constraining all generic virtual registers to
  /// register classes.
  Error addGlobalInstructionSelect() {
    return make_error<StringError>(
        "addGlobalInstructionSelect is not overridden",
        inconvertibleErrorCode());
  }
  /// @}}

  /// High level function that adds all passes necessary to go from llvm IR
  /// representation to the MI representation.
  /// Adds IR based lowering and target specific optimization passes and finally
  /// the core instruction selection passes.
  void addISelPasses();

  /// Add the actual instruction selection passes. This does not include
  /// preparation passes on IR.
  Error addCoreISelPasses();

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  Error addMachinePasses();

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions();

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  void addIRPasses();

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  void addCodeGenPrepare();

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  void addISelPrepare();

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  void addMachineSSAOptimization();

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  Error addFastRegAlloc();

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// LLVMTargetMachine provides standard regalloc passes for most targets.
  void addOptimizedRegAlloc();

  /// Add passes that optimize machine instructions after register allocation.
  void addMachineLateOptimization();

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  void addGCPasses() {}

  /// Add standard basic block placement passes.
  void addBlockPlacement();

  using CreateMCStreamer =
      std::function<Expected<std::unique_ptr<MCStreamer>>(MCContext &)>;
  void addAsmPrinter(CreateMCStreamer) {
    llvm_unreachable("addAsmPrinter is not overridden");
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  void addTargetRegisterAllocator(bool Optimized);

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  void addRegAllocPass(bool Optimized);

  /// Add core register alloator passes which do the actual register assignment
  /// and rewriting. \returns true if any passes were added.
  Error addRegAssignmentFast();
  Error addRegAssignmentOptimized();

  /// Merge all pass manager into one ModulePassManager
  void mergePassManager();

  /// Allow the target to disable a specific pass by default.
  /// Backend can declare unwanted passes in constructor.
  template <typename PassT> void disablePass(unsigned InstanceNum = 0) {
    BeforeCallbacks.emplace_back(
        [Cnt = 0u, InstanceNum](StringRef Name) mutable {
          if (!InstanceNum)
            return PassT::name() != Name;
          if (PassT::name() == Name)
            return ++Cnt == InstanceNum;
          return true;
        });
  }
  template <typename PassT1, typename PassT2, typename... PassTs>
  void disablePass() {
    BeforeCallbacks.emplace_back([](StringRef Name) {
      return Name != PassT1::name() && Name != PassT2::name() &&
             ((Name != PassTs::name()) && ...);
    });
  }

  /// Insert InsertedPass pass after TargetPass pass.
  /// Only machine function passes are supported.
  template <typename TargetPassT, typename InsertedPassT>
  void insertPass(InsertedPassT &&Pass, unsigned InstanceNum = 0) {
    AfterCallbacks.emplace_back([&, Cnt = 0](StringRef Name) mutable {
      if (Name == TargetPassT::name()) {
        if (!InstanceNum) {
          addPass(std::forward<InsertedPassT>(Pass));
          return;
        }
        if (++Cnt == InstanceNum)
          addPass(std::forward<InsertedPassT>(Pass));
      }
    });
  }

  /// Get internal ModulePassManager, only for test!
  ModulePassManager &getMPM() { return MPM; }

private:
  DerivedT &derived() { return static_cast<DerivedT &>(*this); }

  /// A monotonic stack based method to add pass.
  template <typename PassT> void addPassImpl(PassT &&Pass);

  bool runBeforeAdding(StringRef Name) {
    bool ShouldAdd = true;
    for (auto &C : BeforeCallbacks)
      ShouldAdd &= C(Name);
    return ShouldAdd;
  }

  void runAfterAdding(StringRef Name) {
    for (auto &C : AfterCallbacks)
      C(Name);
  }

  void setStartStopPasses(const TargetPassConfig::StartStopInfo &Info);

  Error verifyStartStop(const TargetPassConfig::StartStopInfo &Info) const;

  SmallVector<llvm::unique_function<bool(StringRef)>, 4> BeforeCallbacks;
  SmallVector<llvm::unique_function<void(StringRef)>, 4> AfterCallbacks;

  /// Helper variable for `-start-before/-start-after/-stop-before/-stop-after`
  bool Started = true;
  bool Stopped = true;

  enum class PassType {
    ModulePass,
    FunctionPass,
    MachineFunctionPass,
  };

  std::stack<PassType> MonoStack;
  ModulePassManager MPM;
  FunctionPassManager FPM;
  MachineFunctionPassManager MFPM;
};

template <typename Derived, typename TargetMachineT>
template <typename PassT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addPassImpl(PassT &&Pass) {
  static_assert((is_module_pass_v<PassT> || is_function_pass_v<PassT> ||
                 is_machine_function_pass_v<PassT>)&&"Unexpected pass type!");

  constexpr PassType PT = is_module_pass_v<PassT> ? PassType::ModulePass
                          : is_function_pass_v<PassT>
                              ? PassType::FunctionPass
                              : PassType::MachineFunctionPass;

  while (!MonoStack.empty() && MonoStack.top() > PT) {
    switch (MonoStack.top()) {
    case PassType::MachineFunctionPass:
      FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
      MFPM = MachineFunctionPassManager();
      break;
    case PassType::FunctionPass:
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      FPM = FunctionPassManager();
      break;
    case PassType::ModulePass:
      llvm_unreachable("Unexpected pass type!");
    }
    MonoStack.pop();
  }
  if (MonoStack.empty() || MonoStack.top() < PT)
    MonoStack.push(PT);

  if constexpr (PT == PassType::ModulePass)
    MPM.addPass(std::forward<PassT>(Pass));
  else if constexpr (PT == PassType::FunctionPass)
    FPM.addPass(std::forward<PassT>(Pass));
  else
    MFPM.addPass(std::forward<PassT>(Pass));
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::mergePassManager() {
  if (MonoStack.empty())
    return;

  switch (MonoStack.top()) {
  case PassType::MachineFunctionPass:
    FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
    [[fallthrough]];
  case PassType::FunctionPass:
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    [[fallthrough]];
  case PassType::ModulePass:
    break;
  }
  MFPM = MachineFunctionPassManager();
  FPM = FunctionPassManager();
  MonoStack = decltype(MonoStack)();
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::buildPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType) {
  auto StartStopInfo = TargetPassConfig::getStartStopInfo(*PIC);
  if (!StartStopInfo)
    return StartStopInfo.takeError();
  setStartStopPasses(*StartStopInfo);

  bool PrintAsm = TargetPassConfig::willCompleteCodeGenPipeline();
  bool PrintMIR = !PrintAsm && FileType != CodeGenFileType::Null;

  addPass<RequireAnalysisPass<MachineModuleAnalysis, Module>,
          RequireAnalysisPass<ProfileSummaryAnalysis, Module>,
          RequireAnalysisPass<CollectorMetadataAnalysis, Module>>();
  addISelPasses();

  if (PrintMIR)
    addPass(PrintMIRPreparePass(Out));

  if (auto Err = addCoreISelPasses())
    return Err;

  if (auto Err = derived().addMachinePasses())
    return Err;

  if (PrintAsm) {
    derived().addAsmPrinter([this, &Out, DwoOut, FileType](MCContext &Ctx) {
      return this->TM.createMCStreamer(Out, DwoOut, FileType, Ctx);
    });
  }

  if (PrintMIR)
    addPass(PrintMIRPass(Out));

  addPass(InvalidateAnalysisPass<MachineFunctionAnalysis>());
  return verifyStartStop(*StartStopInfo);
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::setStartStopPasses(
    const TargetPassConfig::StartStopInfo &Info) {
  if (!Info.StartPass.empty()) {
    Started = false;
    BeforeCallbacks.emplace_back([this, &Info, AfterFlag = Info.StartAfter,
                                  Count = 0u](StringRef ClassName) mutable {
      if (Count == Info.StartInstanceNum) {
        if (AfterFlag) {
          AfterFlag = false;
          Started = true;
        }
        return Started;
      }

      auto PassName = PIC->getPassNameForClassName(ClassName);
      if (Info.StartPass == PassName && ++Count == Info.StartInstanceNum)
        Started = !Info.StartAfter;

      return Started;
    });
  }

  if (!Info.StopPass.empty()) {
    Stopped = false;
    BeforeCallbacks.emplace_back([this, &Info, AfterFlag = Info.StopAfter,
                                  Count = 0u](StringRef ClassName) mutable {
      if (Count == Info.StopInstanceNum) {
        if (AfterFlag) {
          AfterFlag = false;
          Stopped = true;
        }
        return !Stopped;
      }

      auto PassName = PIC->getPassNameForClassName(ClassName);
      if (Info.StopPass == PassName && ++Count == Info.StopInstanceNum)
        Stopped = !Info.StopAfter;
      return !Stopped;
    });
  }
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::verifyStartStop(
    const TargetPassConfig::StartStopInfo &Info) const {
  if (Started && Stopped)
    return Error::success();

  if (!Started)
    return make_error<StringError>(
        "Can't find start pass \"" +
            PIC->getPassNameForClassName(Info.StartPass) + "\".",
        std::make_error_code(std::errc::invalid_argument));
  if (!Stopped)
    return make_error<StringError>(
        "Can't find stop pass \"" +
            PIC->getPassNameForClassName(Info.StopPass) + "\".",
        std::make_error_code(std::errc::invalid_argument));
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPasses() {
  derived().addGlobalMergePass();
  if (TM.useEmulatedTLS())
    addPass(LowerEmuTLSPass());

  addPass(PreISelIntrinsicLoweringPass(TM));

  derived().addIRPasses();
  derived().addCodeGenPrepare();
  addPassesToHandleExceptions();
  derived().addISelPrepare();
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addIRPasses() {
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!Opt.DisableVerify)
    addPass(VerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableLSR) {
    addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass(),
                                            /*UseMemorySSA=*/true));
    // FIXME: use -stop-after so we could remove PrintLSR
    if (Opt.PrintLSR)
      addPass(PrintFunctionPass(dbgs(), "\n\n*** Code after LSR ***\n"));
  }

  if (getOptLevel() != CodeGenOptLevel::None) {
    // The MergeICmpsPass tries to create memcmp calls by grouping sequences of
    // loads and compares. ExpandMemCmpPass then tries to expand those calls
    // into optimally-sized loads and compares. The transforms are enabled by a
    // target lowering hook.
    if (!Opt.DisableMergeICmps)
      addPass(MergeICmpsPass());
    addPass(ExpandMemCmpPass(&TM));
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addPass(GCLoweringPass(), ShadowStackGCLoweringPass(),
          LowerConstantIntrinsicsPass());

  // Make sure that no unreachable blocks are instruction selected.
  addPass(UnreachableBlockElimPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableConstantHoisting)
    addPass(ConstantHoistingPass());

  // Replace calls to LLVM intrinsics (e.g., exp, log) operating on vector
  // operands with calls to the corresponding functions in a vector library.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(ReplaceWithVeclib());

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opt.DisablePartialLibcallInlining)
    addPass(PartiallyInlineLibCallsPass());

  // Instrument function entry and exit, e.g. with calls to mcount().
  addPass(EntryExitInstrumenterPass(/*PostInlining=*/true));

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  addPass(ScalarizeMaskedMemIntrinPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  addPass(ExpandReductionsPass());

  // Convert conditional moves to conditional jumps when profitable.
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableSelectOptimize)
    addPass(SelectOptimizePass(&TM));
}

/// Turn exception handling constructs into something the code generators can
/// handle.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived,
                        TargetMachineT>::addPassesToHandleExceptions() {
  const MCAsmInfo *MCAI = TM.getMCAsmInfo();
  assert(MCAI && "No MCAsmInfo");
  switch (MCAI->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    addPass(SjLjEHPreparePass(&TM));
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    addPass(DwarfEHPreparePass(&TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addPass(WinEHPreparePass(), DwarfEHPreparePass(&TM));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there.
    addPass(WinEHPreparePass(/*DemoteCatchSwitchPHIOnly=*/false),
            WasmEHPreparePass());
    break;
  case ExceptionHandling::None:
    addPass(LowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    addPass(UnreachableBlockElimPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None && !Opt.DisableCGP)
    addPass(CodeGenPreparePass(&TM));
  // TODO: Default ctor'd RewriteSymbolPass is no-op.
  // addPass(RewriteSymbolPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addISelPrepare() {
  derived().addPreISel();

  addPass(CallBrPreparePass());
  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addPass(SafeStackPass(&TM), StackProtectorPass(&TM));

  if (Opt.PrintISelInput)
    addPass(PrintFunctionPass(dbgs(),
                              "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!Opt.DisableVerify)
    addPass(VerifierPass());
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addCoreISelPasses() {
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM.setO0WantsFastISel(Opt.EnableFastISelOption.value_or(true));

  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  if (Opt.EnableFastISelOption && *Opt.EnableFastISelOption == true)
    Selector = SelectorType::FastISel;
  else if ((Opt.EnableGlobalISelOption &&
            *Opt.EnableGlobalISelOption == true) ||
           (TM.Options.EnableGlobalISel &&
            (!Opt.EnableGlobalISelOption ||
             *Opt.EnableGlobalISelOption == false)))
    Selector = SelectorType::GlobalISel;
  else if (TM.getOptLevel() == CodeGenOptLevel::None && TM.getO0WantsFastISel())
    Selector = SelectorType::FastISel;
  else
    Selector = SelectorType::SelectionDAG;

  // Set consistently TM.Options.EnableFastISel and EnableGlobalISel.
  if (Selector == SelectorType::FastISel) {
    TM.setFastISel(true);
    TM.setGlobalISel(false);
  } else if (Selector == SelectorType::GlobalISel) {
    TM.setFastISel(false);
    TM.setGlobalISel(true);
  }

  // Add instruction selector passes.
  if (Selector == SelectorType::GlobalISel) {
    if (auto Err = derived().addIRTranslator())
      return Err;

    derived().addPreLegalizeMachineIR();

    if (auto Err = derived().addLegalizeMachineIR())
      return Err;

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    derived().addPreRegBankSelect();

    if (auto Err = derived().addRegBankSelect())
      return Err;

    derived().addPreGlobalInstructionSelect();

    if (auto Err = derived().addGlobalInstructionSelect())
      return Err;

    // Pass to reset the MachineFunction if the ISel failed.
    addPass(ResetMachineFunctionPass(reportDiagnosticWhenGlobalISelFallback(),
                                     isGlobalISelAbortEnabled()));

    // Provide a fallback path when we do not want to abort on
    // not-yet-supported input.
    if (!isGlobalISelAbortEnabled())
      if (auto Err = derived().addInstSelector())
        return Err;

  } else if (auto Err = derived().addInstSelector())
    return Err;

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  addPass(FinalizeISelPass());

  // // Print the instruction selected machine code...
  // printAndVerify("After Instruction Selection");

  return Error::success();
}

/// Add the complete set of target-independent postISel code generator passes.
///
/// This can be read as the standard order of major LLVM CodeGen stages. Stages
/// with nontrivial configuration or multiple passes are broken out below in
/// add%Stage routines.
///
/// Any CodeGenPassBuilder<Derived, TargetMachine>::addXX routine may be
/// overriden by the Target. The addPre/Post methods with empty header
/// implementations allow injecting target-specific fixups just before or after
/// major stages. Additionally, targets have the flexibility to change pass
/// order within a stage by overriding default implementation of add%Stage
/// routines below. Each technique has maintainability tradeoffs because
/// alternate pass orders are not well supported. addPre/Post works better if
/// the target pass is easily tied to a common pass. But if it has subtle
/// dependencies on multiple passes, the target should override the stage
/// instead.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addMachinePasses() {
  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    derived().addMachineSSAOptimization();
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(LocalStackSlotPass());
  }

  if (TM.Options.EnableIPRA)
    addPass(RegUsageInfoPropagationPass());

  // Run pre-ra passes.
  derived().addPreRegAlloc();

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (*Opt.OptimizeRegAlloc) {
    derived().addOptimizedRegAlloc();
  } else {
    if (auto Err = derived().addFastRegAlloc())
      return Err;
  }

  // Run post-ra passes.
  derived().addPostRegAlloc();

  addPass(RemoveRedundantDebugValuesPass());

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass<PostRAMachineSinkingPass, ShrinkWrapPass>();

  addPass(PrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  addPass(ExpandPostRAPseudosPass());

  // Run pre-sched2 passes.
  derived().addPreSched2();

  if (Opt.EnableImplicitNullChecks)
    addPass(ImplicitNullChecksPass());

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOptLevel::None &&
      !TM.targetSchedulesPostRAScheduling()) {
    if (Opt.MISchedPostRA)
      addPass(PostMachineSchedulerPass());
    else
      addPass(PostRASchedulerPass());
  }

  // GC
  derived().addGCPasses();

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    derived().addBlockPlacement();

  // Insert before XRay Instrumentation.
  addPass<FEntryInserterPass, XRayInstrumentationPass, PatchableFunctionPass>();

  derived().addPreEmitPass();

  if (TM.Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    addPass(RegUsageInfoCollectorPass());

  addPass<FuncletLayoutPass, StackMapLivenessPass, LiveDebugValuesPass,
          MachineSanitizerBinaryMetadata>();

  if (TM.Options.EnableMachineOutliner &&
      getOptLevel() != CodeGenOptLevel::None &&
      Opt.EnableMachineOutliner != RunOutliner::NeverOutline) {
    bool RunOnAllFunctions =
        (Opt.EnableMachineOutliner == RunOutliner::AlwaysOutline);
    bool AddOutliner = RunOnAllFunctions || TM.Options.SupportsDefaultOutlining;
    if (AddOutliner)
      addPass(MachineOutlinerPass(RunOnAllFunctions));
  }

  // Add passes that directly emit MI after all other MI passes.
  derived().addPreEmitPass2();

  return Error::success();
}

/// Add passes that optimize machine instructions in SSA form.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  addPass(EarlyTailDuplicatePass());

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(OptimizePHIsPass());

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(StackColoringPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(LocalStackSlotPass());

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(DeadMachineInstructionElimPass());

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  derived().addILPOpts();

  addPass(EarlyMachineLICMPass());
  addPass(MachineCSEPass());

  addPass(MachineSinkingPass());

  addPass(PeepholeOptimizerPass());
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addPass(DeadMachineInstructionElimPass());
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

/// Instantiate the default register allocator pass for this target for either
/// the optimized or unoptimized allocation path. This will be added to the pass
/// manager by addFastRegAlloc in the unoptimized case or addOptimizedRegAlloc
/// in the optimized case.
///
/// A target that uses the standard regalloc pass order for fast or optimized
/// allocation may still override this for per-target regalloc
/// selection. But -regalloc=... always takes precedence.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addTargetRegisterAllocator(
    bool Optimized) {
  if (Optimized)
    addPass(RAGreedyPass());
  else
    addPass(RAFastPass());
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addRegAllocPass(
    bool Optimized) {
  // TODO: Parse Opt.RegAlloc to add register allocator.
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentFast() {
  // TODO: Ensure allocator is default or fast.
  addRegAllocPass(false);
  return Error::success();
}

template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addRegAssignmentOptimized() {
  // Add the selected register allocation pass.
  addRegAllocPass(true);

  // Allow targets to change the register assignments before rewriting.
  derived().addPreRewrite();

  // Finally rewrite virtual registers.
  addPass(VirtRegRewriterPass());
  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addPass(StackSlotColoringPass());

  return Error::success();
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
template <typename Derived, typename TargetMachineT>
Error CodeGenPassBuilder<Derived, TargetMachineT>::addFastRegAlloc() {
  addPass(PHIEliminationPass());
  addPass(TwoAddressInstructionPass());
  return derived().addRegAssignmentFast();
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addOptimizedRegAlloc() {
  addPass(DetectDeadLanesPass());

  addPass(InitUndefPass());

  addPass(ProcessImplicitDefsPass());

  // Edge splitting is smarter with machine loop info.
  addPass(PHIEliminationPass());

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (Opt.EarlyLiveIntervals)
    addPass(LiveIntervalsPass());

  addPass(TwoAddressInstructionPass());
  addPass(RegisterCoalescerPass());

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  addPass(RenameIndependentSubregsPass());

  // PreRA instruction scheduling.
  addPass(MachineSchedulerPass());

  if (derived().addRegAssignmentOptimized()) {
    // Allow targets to expand pseudo instructions depending on the choice of
    // registers before MachineCopyPropagation.
    derived().addPostRewrite();

    // Copy propagate to forward register uses and try to eliminate COPYs that
    // were not coalesced.
    addPass(MachineCopyPropagationPass());

    // Run post-ra machine LICM to hoist reloads / remats.
    //
    // FIXME: can this move into MachineLateOptimization?
    addPass(MachineLICMPass());
  }
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addMachineLateOptimization() {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addPass(BranchFolderPass());

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM.requiresStructuredCFG())
    addPass(TailDuplicatePass());

  // Cleanup of redundant (identical) address/immediate loads.
  addPass(MachineLateInstrsCleanupPass());

  // Copy propagation.
  addPass(MachineCopyPropagationPass());
}

/// Add standard basic block placement passes.
template <typename Derived, typename TargetMachineT>
void CodeGenPassBuilder<Derived, TargetMachineT>::addBlockPlacement() {
  addPass(MachineBlockPlacementPass());
  // Run a separate pass to collect block placement statistics.
  if (Opt.EnableBlockPlacementStats)
    addPass(MachineBlockPlacementStatsPass());
}

} // namespace llvm

#endif // LLVM_PASSES_CODEGENPASSBUILDER_H
