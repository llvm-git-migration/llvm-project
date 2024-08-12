//===------ ELFNixPlatform.cpp - Utilities for executing ELFNix in Orc -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/ELF_x86_64.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/ExecutionEngine/JITLink/ppc64.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

class DSOHandleMaterializationUnit : public MaterializationUnit {
public:
  DSOHandleMaterializationUnit(ELFNixPlatform &ENP,
                               const SymbolStringPtr &DSOHandleSymbol)
      : MaterializationUnit(
            createDSOHandleSectionInterface(ENP, DSOHandleSymbol)),
        ENP(ENP) {}

  StringRef getName() const override { return "DSOHandleMU"; }

  void materialize(std::unique_ptr<MaterializationResponsibility> R) override {
    unsigned PointerSize;
    llvm::endianness Endianness;
    jitlink::Edge::Kind EdgeKind;
    const auto &TT = ENP.getExecutionSession().getTargetTriple();

    switch (TT.getArch()) {
    case Triple::x86_64:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::x86_64::Pointer64;
      break;
    case Triple::aarch64:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::aarch64::Pointer64;
      break;
    case Triple::ppc64:
      PointerSize = 8;
      Endianness = llvm::endianness::big;
      EdgeKind = jitlink::ppc64::Pointer64;
      break;
    case Triple::ppc64le:
      PointerSize = 8;
      Endianness = llvm::endianness::little;
      EdgeKind = jitlink::ppc64::Pointer64;
      break;
    default:
      llvm_unreachable("Unrecognized architecture");
    }

    // void *__dso_handle = &__dso_handle;
    auto G = std::make_unique<jitlink::LinkGraph>(
        "<DSOHandleMU>", TT, PointerSize, Endianness,
        jitlink::getGenericEdgeKindName);
    auto &DSOHandleSection =
        G->createSection(".data.__dso_handle", MemProt::Read);
    auto &DSOHandleBlock = G->createContentBlock(
        DSOHandleSection, getDSOHandleContent(PointerSize), orc::ExecutorAddr(),
        8, 0);
    auto &DSOHandleSymbol = G->addDefinedSymbol(
        DSOHandleBlock, 0, *R->getInitializerSymbol(), DSOHandleBlock.getSize(),
        jitlink::Linkage::Strong, jitlink::Scope::Default, false, true);
    DSOHandleBlock.addEdge(EdgeKind, 0, DSOHandleSymbol, 0);

    ENP.getObjectLinkingLayer().emit(std::move(R), std::move(G));
  }

  void discard(const JITDylib &JD, const SymbolStringPtr &Sym) override {}

private:
  static MaterializationUnit::Interface
  createDSOHandleSectionInterface(ELFNixPlatform &ENP,
                                  const SymbolStringPtr &DSOHandleSymbol) {
    SymbolFlagsMap SymbolFlags;
    SymbolFlags[DSOHandleSymbol] = JITSymbolFlags::Exported;
    return MaterializationUnit::Interface(std::move(SymbolFlags),
                                          DSOHandleSymbol);
  }

  ArrayRef<char> getDSOHandleContent(size_t PointerSize) {
    static const char Content[8] = {0};
    assert(PointerSize <= sizeof Content);
    return {Content, PointerSize};
  }

  ELFNixPlatform &ENP;
};

} // end anonymous namespace

namespace llvm {
namespace orc {

Expected<std::unique_ptr<ELFNixPlatform>> ELFNixPlatform::Create(
    ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
    JITDylib &PlatformJD, std::unique_ptr<DefinitionGenerator> OrcRuntime,
    std::optional<SymbolAliasMap> RuntimeAliases) {

  // If the target is not supported then bail out immediately.
  if (!supportedTarget(ES.getTargetTriple()))
    return make_error<StringError>("Unsupported ELFNixPlatform triple: " +
                                       ES.getTargetTriple().str(),
                                   inconvertibleErrorCode());

  auto &EPC = ES.getExecutorProcessControl();

  // Create default aliases if the caller didn't supply any.
  if (!RuntimeAliases) {
    auto StandardRuntimeAliases = standardPlatformAliases(ES, PlatformJD);
    if (!StandardRuntimeAliases)
      return StandardRuntimeAliases.takeError();
    RuntimeAliases = std::move(*StandardRuntimeAliases);
  }

  // Define the aliases.
  if (auto Err = PlatformJD.define(symbolAliases(std::move(*RuntimeAliases))))
    return std::move(Err);

  // Add JIT-dispatch function support symbols.
  if (auto Err = PlatformJD.define(
          absoluteSymbols({{ES.intern("__orc_rt_jit_dispatch"),
                            {EPC.getJITDispatchInfo().JITDispatchFunction,
                             JITSymbolFlags::Exported}},
                           {ES.intern("__orc_rt_jit_dispatch_ctx"),
                            {EPC.getJITDispatchInfo().JITDispatchContext,
                             JITSymbolFlags::Exported}}})))
    return std::move(Err);

  // Create the instance.
  Error Err = Error::success();
  auto P = std::unique_ptr<ELFNixPlatform>(new ELFNixPlatform(
      ES, ObjLinkingLayer, PlatformJD, std::move(OrcRuntime), Err));
  if (Err)
    return std::move(Err);
  return std::move(P);
}

Expected<std::unique_ptr<ELFNixPlatform>>
ELFNixPlatform::Create(ExecutionSession &ES,
                       ObjectLinkingLayer &ObjLinkingLayer,
                       JITDylib &PlatformJD, const char *OrcRuntimePath,
                       std::optional<SymbolAliasMap> RuntimeAliases) {

  // Create a generator for the ORC runtime archive.
  auto OrcRuntimeArchiveGenerator =
      StaticLibraryDefinitionGenerator::Load(ObjLinkingLayer, OrcRuntimePath);
  if (!OrcRuntimeArchiveGenerator)
    return OrcRuntimeArchiveGenerator.takeError();

  return Create(ES, ObjLinkingLayer, PlatformJD,
                std::move(*OrcRuntimeArchiveGenerator),
                std::move(RuntimeAliases));
}

Error ELFNixPlatform::setupJITDylib(JITDylib &JD) {
  if (auto Err = JD.define(std::make_unique<DSOHandleMaterializationUnit>(
          *this, DSOHandleSymbol)))
    return Err;

  return ES.lookup({&JD}, DSOHandleSymbol).takeError();
}

Error ELFNixPlatform::teardownJITDylib(JITDylib &JD) {
  std::lock_guard<std::mutex> Lock(PlatformMutex);
  auto I = JITDylibToHandleAddr.find(&JD);
  if (I != JITDylibToHandleAddr.end()) {
    assert(HandleAddrToJITDylib.count(I->second) &&
           "HandleAddrToJITDylib missing entry");
    HandleAddrToJITDylib.erase(I->second);
    JITDylibToHandleAddr.erase(I);
  }
  return Error::success();
}

Error ELFNixPlatform::notifyAdding(ResourceTracker &RT,
                                   const MaterializationUnit &MU) {

  auto &JD = RT.getJITDylib();
  const auto &InitSym = MU.getInitializerSymbol();
  if (!InitSym)
    return Error::success();

  RegisteredInitSymbols[&JD].add(InitSym,
                                 SymbolLookupFlags::WeaklyReferencedSymbol);
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform: Registered init symbol " << *InitSym
           << " for MU " << MU.getName() << "\n";
  });
  return Error::success();
}

Error ELFNixPlatform::notifyRemoving(ResourceTracker &RT) {
  llvm_unreachable("Not supported yet");
}

static void addAliases(ExecutionSession &ES, SymbolAliasMap &Aliases,
                       ArrayRef<std::pair<const char *, const char *>> AL) {
  for (auto &KV : AL) {
    auto AliasName = ES.intern(KV.first);
    assert(!Aliases.count(AliasName) && "Duplicate symbol name in alias map");
    Aliases[std::move(AliasName)] = {ES.intern(KV.second),
                                     JITSymbolFlags::Exported};
  }
}

Expected<SymbolAliasMap>
ELFNixPlatform::standardPlatformAliases(ExecutionSession &ES,
                                        JITDylib &PlatformJD) {
  SymbolAliasMap Aliases;
  addAliases(ES, Aliases, requiredCXXAliases());
  addAliases(ES, Aliases, standardRuntimeUtilityAliases());
  return Aliases;
}

ArrayRef<std::pair<const char *, const char *>>
ELFNixPlatform::requiredCXXAliases() {
  static const std::pair<const char *, const char *> RequiredCXXAliases[] = {
      {"__cxa_atexit", "__orc_rt_elfnix_cxa_atexit"},
      {"atexit", "__orc_rt_elfnix_atexit"}};

  return ArrayRef<std::pair<const char *, const char *>>(RequiredCXXAliases);
}

ArrayRef<std::pair<const char *, const char *>>
ELFNixPlatform::standardRuntimeUtilityAliases() {
  static const std::pair<const char *, const char *>
      StandardRuntimeUtilityAliases[] = {
          {"__orc_rt_run_program", "__orc_rt_elfnix_run_program"},
          {"__orc_rt_jit_dlerror", "__orc_rt_elfnix_jit_dlerror"},
          {"__orc_rt_jit_dlopen", "__orc_rt_elfnix_jit_dlopen"},
          {"__orc_rt_jit_dlclose", "__orc_rt_elfnix_jit_dlclose"},
          {"__orc_rt_jit_dlsym", "__orc_rt_elfnix_jit_dlsym"},
          {"__orc_rt_log_error", "__orc_rt_log_error_to_stderr"}};

  return ArrayRef<std::pair<const char *, const char *>>(
      StandardRuntimeUtilityAliases);
}

bool ELFNixPlatform::supportedTarget(const Triple &TT) {
  switch (TT.getArch()) {
  case Triple::x86_64:
  case Triple::aarch64:
  // FIXME: jitlink for ppc64 hasn't been well tested, leave it unsupported
  // right now.
  case Triple::ppc64le:
    return true;
  default:
    return false;
  }
}

ELFNixPlatform::ELFNixPlatform(
    ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
    JITDylib &PlatformJD,
    std::unique_ptr<DefinitionGenerator> OrcRuntimeGenerator, Error &Err)
    : ES(ES), ObjLinkingLayer(ObjLinkingLayer),
      DSOHandleSymbol(ES.intern("__dso_handle")) {
  ErrorAsOutParameter _(&Err);
  ObjLinkingLayer.addPlugin(std::make_unique<ELFNixPlatformPlugin>(*this));

  PlatformJD.addGenerator(std::move(OrcRuntimeGenerator));

  // PlatformJD hasn't been 'set-up' by the platform yet (since we're creating
  // the platform now), so set it up.
  if (auto E2 =
          PlatformJD.define(std::make_unique<DSOHandleMaterializationUnit>(
              *this, DSOHandleSymbol))) {
    Err = std::move(E2);
    return;
  }

  auto E = ES.lookup({&PlatformJD}, DSOHandleSymbol);
  if (auto E2 = E.takeError()) {
    Err = std::move(E2);
    return;
  }
  DSOHandleAddr = E->getAddress();
  // RegisteredInitSymbols[&PlatformJD].add(
  //   DSOHandleSymbol, SymbolLookupFlags::WeaklyReferencedSymbol);

  // Associate wrapper function tags with JIT-side function implementations.
  if (auto E2 = associateRuntimeSupportFunctions(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  // Lookup addresses of runtime functions callable by the platform,
  // call the platform bootstrap function to initialize the platform-state
  // object in the executor.
  if (auto E2 = bootstrapELFNixRuntime(PlatformJD)) {
    Err = std::move(E2);
    return;
  }

  JDBootstrapStates.clear();
}

Error ELFNixPlatform::associateRuntimeSupportFunctions(JITDylib &PlatformJD) {
  ExecutionSession::JITDispatchHandlerAssociationMap WFs;

  using RecordInitializersSPSSig =
      SPSExpected<SPSELFNixJITDylibDepInfoMap>(SPSExecutorAddr);
  WFs[ES.intern("__orc_rt_elfnix_push_initializers_tag")] =
      ES.wrapAsyncWithSPS<RecordInitializersSPSSig>(
          this, &ELFNixPlatform::rt_recordInitializers);

  using LookupSymbolSPSSig =
      SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString);
  WFs[ES.intern("__orc_rt_elfnix_symbol_lookup_tag")] =
      ES.wrapAsyncWithSPS<LookupSymbolSPSSig>(this,
                                              &ELFNixPlatform::rt_lookupSymbol);

  return ES.registerJITDispatchHandlers(PlatformJD, std::move(WFs));
}

void ELFNixPlatform::pushInitializersLoop(
    PushInitializersSendResultFn SendResult, JITDylibSP JD) {
  DenseMap<JITDylib *, SymbolLookupSet> NewInitSymbols;
  DenseMap<JITDylib *, SmallVector<JITDylib *>> JDDepMap;
  SmallVector<JITDylib *, 16> Worklist({JD.get()});

  ES.runSessionLocked([&]() {
    while (!Worklist.empty()) {
      // FIXME: Check for defunct dylibs.

      auto DepJD = Worklist.back();
      Worklist.pop_back();

      // If we've already visited this JITDylib on this iteration then continue.
      if (JDDepMap.count(DepJD))
        continue;

      // Add dep info.
      auto &DM = JDDepMap[DepJD];
      DepJD->withLinkOrderDo([&](const JITDylibSearchOrder &O) {
        for (auto &KV : O) {
          if (KV.first == DepJD)
            continue;
          DM.push_back(KV.first);
          Worklist.push_back(KV.first);
        }
      });

      // Add any registered init symbols.
      auto RISItr = RegisteredInitSymbols.find(DepJD);
      if (RISItr != RegisteredInitSymbols.end()) {
        NewInitSymbols[DepJD] = std::move(RISItr->second);
        RegisteredInitSymbols.erase(RISItr);
      }
    }
  });

  // If there are no further init symbols to look up then send the link order
  // (as a list of header addresses) to the caller.
  if (NewInitSymbols.empty()) {

    // To make the list intelligible to the runtime we need to convert all
    // JITDylib pointers to their header addresses. Only include JITDylibs
    // that appear in the JITDylibToHandleAddr map (i.e. those that have been
    // through setupJITDylib) -- bare JITDylibs aren't managed by the platform.
    DenseMap<JITDylib *, ExecutorAddr> HeaderAddrs;
    HeaderAddrs.reserve(JDDepMap.size());
    {
      std::lock_guard<std::mutex> Lock(PlatformMutex);
      for (auto &KV : JDDepMap) {
        auto I = JITDylibToHandleAddr.find(KV.first);
        if (I != JITDylibToHandleAddr.end())
          HeaderAddrs[KV.first] = I->second;
      }
    }

    // Build the dep info map to return.
    ELFNixJITDylibDepInfoMap DIM;
    DIM.reserve(JDDepMap.size());
    for (auto &KV : JDDepMap) {
      auto HI = HeaderAddrs.find(KV.first);
      // Skip unmanaged JITDylibs.
      if (HI == HeaderAddrs.end())
        continue;
      auto H = HI->second;
      ELFNixJITDylibDepInfo DepInfo;
      for (auto &Dep : KV.second) {
        auto HJ = HeaderAddrs.find(Dep);
        if (HJ != HeaderAddrs.end())
          DepInfo.push_back(HJ->second);
      }
      DIM.push_back(std::make_pair(H, std::move(DepInfo)));
    }
    SendResult(DIM);
    return;
  }

  // Otherwise issue a lookup and re-run this phase when it completes.
  lookupInitSymbolsAsync(
      [this, SendResult = std::move(SendResult), JD](Error Err) mutable {
        if (Err)
          SendResult(std::move(Err));
        else
          pushInitializersLoop(std::move(SendResult), JD);
      },
      ES, std::move(NewInitSymbols));
}

void ELFNixPlatform::rt_recordInitializers(
    PushInitializersSendResultFn SendResult, ExecutorAddr JDHeaderAddr) {
  JITDylibSP JD;
  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HandleAddrToJITDylib.find(JDHeaderAddr);
    if (I != HandleAddrToJITDylib.end())
      JD = I->second;
  }

  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform::rt_pushInitializers(" << JDHeaderAddr << ") ";
    if (JD)
      dbgs() << "pushing initializers for " << JD->getName() << "\n";
    else
      dbgs() << "No JITDylib for header address.\n";
  });

  if (!JD) {
    SendResult(make_error<StringError>("No JITDylib with header addr " +
                                           formatv("{0:x}", JDHeaderAddr),
                                       inconvertibleErrorCode()));
    return;
  }

  pushInitializersLoop(std::move(SendResult), JD);
}

void ELFNixPlatform::rt_lookupSymbol(SendSymbolAddressFn SendResult,
                                     ExecutorAddr Handle,
                                     StringRef SymbolName) {
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform::rt_lookupSymbol(\"" << Handle << "\")\n";
  });

  JITDylib *JD = nullptr;

  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    auto I = HandleAddrToJITDylib.find(Handle);
    if (I != HandleAddrToJITDylib.end())
      JD = I->second;
  }

  if (!JD) {
    LLVM_DEBUG(dbgs() << "  No JITDylib for handle " << Handle << "\n");
    SendResult(make_error<StringError>("No JITDylib associated with handle " +
                                           formatv("{0:x}", Handle),
                                       inconvertibleErrorCode()));
    return;
  }

  // Use functor class to work around XL build compiler issue on AIX.
  class RtLookupNotifyComplete {
  public:
    RtLookupNotifyComplete(SendSymbolAddressFn &&SendResult)
        : SendResult(std::move(SendResult)) {}
    void operator()(Expected<SymbolMap> Result) {
      if (Result) {
        assert(Result->size() == 1 && "Unexpected result map count");
        SendResult(Result->begin()->second.getAddress());
      } else {
        SendResult(Result.takeError());
      }
    }

  private:
    SendSymbolAddressFn SendResult;
  };

  ES.lookup(
      LookupKind::DLSym, {{JD, JITDylibLookupFlags::MatchExportedSymbolsOnly}},
      SymbolLookupSet(ES.intern(SymbolName)), SymbolState::Ready,
      RtLookupNotifyComplete(std::move(SendResult)), NoDependenciesToRegister);
}

Error ELFNixPlatform::bootstrapELFNixRuntime(JITDylib &PlatformJD) {
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder(&PlatformJD),
          {
              {ES.intern("__orc_rt_elfnix_platform_bootstrap"),
               &orc_rt_elfnix_platform_bootstrap},
              {ES.intern("__orc_rt_elfnix_platform_shutdown"),
               &orc_rt_elfnix_platform_shutdown},
              {ES.intern("__orc_rt_elfnix_register_jitdylib"),
               &orc_rt_elfnix_register_jitdylib},
              {ES.intern("__orc_rt_elfnix_deregister_jitdylib"),
               &orc_rt_elfnix_deregister_jitdylib},
              {ES.intern("__orc_rt_elfnix_register_init_sections"),
               &orc_rt_elfnix_register_init_sections},
              {ES.intern("__orc_rt_elfnix_deregister_init_sections"),
               &orc_rt_elfnix_deregister_init_sections},
              {ES.intern("__orc_rt_elfnix_register_object_sections"),
               &orc_rt_elfnix_register_object_sections},
              {ES.intern("__orc_rt_elfnix_deregister_object_sections"),
               &orc_rt_elfnix_deregister_object_sections},
          }))
    return Err;


  if (auto Err = ES.callSPSWrapper<void(SPSExecutorAddr)>(
          orc_rt_elfnix_platform_bootstrap, DSOHandleAddr))
    return Err;

  // FIXME: Ordering is fuzzy here. We're probably best off saying
  // "behavior is undefined if code that uses the runtime is added before
  // the platform constructor returns", then move all this to the constructor.
  RuntimeBootstrapped = true;
  std::vector<ELFPerObjectSectionsToRegister> DeferredPOSRs;
  {
    std::lock_guard<std::mutex> Lock(PlatformMutex);
    DeferredPOSRs = std::move(BootstrapPOSRs);
  }

  for (auto KV : JDBootstrapStates) {
    auto &JDBState = KV.second;
    if (auto Err = ES.callSPSWrapper<void(SPSString, SPSExecutorAddr)>(
            orc_rt_elfnix_register_jitdylib, JDBState.JDName,
            JDBState.HeaderAddr))
      return Err;
  }

  for (auto &D : DeferredPOSRs)
    if (auto Err = registerPerObjectSections(D))
      return Err;

  for (auto KV : JDBootstrapStates) {
    auto &JDBState = KV.second;
    if (JDBState.Initializers.empty())
      continue;
    if (auto Err = ES.callSPSWrapper<void(SPSExecutorAddr,
                                          SPSSequence<SPSExecutorAddrRange>)>(
            orc_rt_elfnix_register_init_sections, JDBState.HeaderAddr,
            JDBState.Initializers))
      return Err;
  }
  return Error::success();
}

Error ELFNixPlatform::registerPerObjectSections(
    const ELFPerObjectSectionsToRegister &POSR) {

  if (!orc_rt_elfnix_register_object_sections)
    return make_error<StringError>("Attempting to register per-object "
                                   "sections, but runtime support has not "
                                   "been loaded yet",
                                   inconvertibleErrorCode());

  Error ErrResult = Error::success();
  if (auto Err = ES.callSPSWrapper<shared::SPSError(
                     SPSELFPerObjectSectionsToRegister)>(
          orc_rt_elfnix_register_object_sections, ErrResult, POSR))
    return Err;
  return ErrResult;
}

Expected<uint64_t> ELFNixPlatform::createPThreadKey() {
  if (!orc_rt_elfnix_create_pthread_key)
    return make_error<StringError>(
        "Attempting to create pthread key in target, but runtime support has "
        "not been loaded yet",
        inconvertibleErrorCode());

  Expected<uint64_t> Result(0);
  if (auto Err = ES.callSPSWrapper<SPSExpected<uint64_t>(void)>(
          orc_rt_elfnix_create_pthread_key, Result))
    return std::move(Err);
  return Result;
}

void ELFNixPlatform::ELFNixPlatformPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, jitlink::LinkGraph &LG,
    jitlink::PassConfiguration &Config) {

  bool IsBootstrapping = !MP.RuntimeBootstrapped.load();

  if (auto InitializerSymbol = MR.getInitializerSymbol()) {
    // If the initializer symbol is the __dso_handle symbol then just add
    // the DSO handle support passes.
    if (MR.getInitializerSymbol() == MP.DSOHandleSymbol) {
      addDSOHandleSupportPasses(MR, Config, IsBootstrapping);
      // The DSOHandle materialization unit doesn't require any other
      // support, so we can bail out early.
      return;
    }
    Config.PrePrunePasses.push_back(
        [this, &MR](jitlink::LinkGraph &G) -> Error {
          if (auto Err = preserveInitSections(G, MR))
            return Err;
          return Error::success();
        });
  }

  // Add passes for eh-frame and TLV support.
  addEHAndTLVSupportPasses(MR, Config);

  Config.PostFixupPasses.push_back([this, &JD = MR.getTargetJITDylib(),
                                    IsBootstrapping](jitlink::LinkGraph &G) {
    return registerInitSections(G, JD, IsBootstrapping);
  });
}

ObjectLinkingLayer::Plugin::SyntheticSymbolDependenciesMap
ELFNixPlatform::ELFNixPlatformPlugin::getSyntheticSymbolDependencies(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PluginMutex);
  auto I = InitSymbolDeps.find(&MR);
  if (I != InitSymbolDeps.end()) {
    SyntheticSymbolDependenciesMap Result;
    Result[MR.getInitializerSymbol()] = std::move(I->second);
    InitSymbolDeps.erase(&MR);
    return Result;
  }
  return SyntheticSymbolDependenciesMap();
}

void ELFNixPlatform::ELFNixPlatformPlugin::addDSOHandleSupportPasses(
    MaterializationResponsibility &MR, jitlink::PassConfiguration &Config,
    bool IsBootstraping) {

  Config.PostAllocationPasses.push_back([this, &JD = MR.getTargetJITDylib(),
                                         IsBootstraping](
                                            jitlink::LinkGraph &G) -> Error {
    auto I = llvm::find_if(G.defined_symbols(), [this](jitlink::Symbol *Sym) {
      return Sym->getName() == *MP.DSOHandleSymbol;
    });
    assert(I != G.defined_symbols().end() && "Missing DSO handle symbol");
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto HandleAddr = (*I)->getAddress();
      MP.HandleAddrToJITDylib[HandleAddr] = &JD;
      MP.JITDylibToHandleAddr[&JD] = HandleAddr;

      if (!IsBootstraping) {
        G.allocActions().push_back(
            {cantFail(WrapperFunctionCall::Create<
                      SPSArgList<SPSString, SPSExecutorAddr>>(
                 MP.orc_rt_elfnix_register_jitdylib, JD.getName(), HandleAddr)),
             cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
                 MP.orc_rt_elfnix_deregister_jitdylib, HandleAddr))});
      } else {
        // G.allocActions().push_back(
        //   {{},
        //    cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
        //         MP.orc_rt_elfnix_deregister_jitdylib, HandleAddr))});
        JDBootstrapState BState;
        BState.JD = &JD;
        BState.JDName = JD.getName();
        BState.HeaderAddr = HandleAddr;
        MP.JDBootstrapStates.emplace(&JD, BState);
      }
    }
    return Error::success();
  });
}

void ELFNixPlatform::ELFNixPlatformPlugin::addEHAndTLVSupportPasses(
    MaterializationResponsibility &MR, jitlink::PassConfiguration &Config) {

  // Insert TLV lowering at the start of the PostPrunePasses, since we want
  // it to run before GOT/PLT lowering.

  // TODO: Check that before the fixTLVSectionsAndEdges pass, the GOT/PLT build
  // pass has done. Because the TLS descriptor need to be allocate in GOT.
  Config.PostPrunePasses.push_back(
      [this, &JD = MR.getTargetJITDylib()](jitlink::LinkGraph &G) {
        return fixTLVSectionsAndEdges(G, JD);
      });

  // Add a pass to register the final addresses of the eh-frame and TLV sections
  // with the runtime.
  Config.PostFixupPasses.push_back([this](jitlink::LinkGraph &G) -> Error {
    ELFPerObjectSectionsToRegister POSR;

    if (auto *EHFrameSection = G.findSectionByName(ELFEHFrameSectionName)) {
      jitlink::SectionRange R(*EHFrameSection);
      if (!R.empty())
        POSR.EHFrameSection = R.getRange();
    }

    // Get a pointer to the thread data section if there is one. It will be used
    // below.
    jitlink::Section *ThreadDataSection =
        G.findSectionByName(ELFThreadDataSectionName);

    // Handle thread BSS section if there is one.
    if (auto *ThreadBSSSection = G.findSectionByName(ELFThreadBSSSectionName)) {
      // If there's already a thread data section in this graph then merge the
      // thread BSS section content into it, otherwise just treat the thread
      // BSS section as the thread data section.
      if (ThreadDataSection)
        G.mergeSections(*ThreadDataSection, *ThreadBSSSection);
      else
        ThreadDataSection = ThreadBSSSection;
    }

    // Having merged thread BSS (if present) and thread data (if present),
    // record the resulting section range.
    if (ThreadDataSection) {
      jitlink::SectionRange R(*ThreadDataSection);
      if (!R.empty())
        POSR.ThreadDataSection = R.getRange();
    }

    if (POSR.EHFrameSection.Start || POSR.ThreadDataSection.Start) {

      // If we're still bootstrapping the runtime then just record this
      // frame for now.
      if (!MP.RuntimeBootstrapped) {
        std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
        MP.BootstrapPOSRs.push_back(POSR);
        return Error::success();
      }

      // Otherwise register it immediately.
      if (auto Err = MP.registerPerObjectSections(POSR))
        return Err;
    }

    return Error::success();
  });
}

Error ELFNixPlatform::ELFNixPlatformPlugin::preserveInitSections(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  JITLinkSymbolSet InitSectionSymbols;
  for (auto &InitSection : G.sections()) {
    // Skip non-init sections.
    if (!isELFInitializerSection(InitSection.getName()))
      continue;

    // Make a pass over live symbols in the section: those blocks are already
    // preserved.
    DenseSet<jitlink::Block *> AlreadyLiveBlocks;
    for (auto &Sym : InitSection.symbols()) {
      auto &B = Sym->getBlock();
      if (Sym->isLive() && Sym->getOffset() == 0 &&
          Sym->getSize() == B.getSize() && !AlreadyLiveBlocks.count(&B)) {
        InitSectionSymbols.insert(Sym);
        AlreadyLiveBlocks.insert(&B);
      }
    }

    // Add anonymous symbols to preserve any not-already-preserved blocks.
    for (auto *B : InitSection.blocks())
      if (!AlreadyLiveBlocks.count(B))
        InitSectionSymbols.insert(
            &G.addAnonymousSymbol(*B, 0, B->getSize(), false, true));
  }

  if (!InitSectionSymbols.empty()) {
    std::lock_guard<std::mutex> Lock(PluginMutex);
    InitSymbolDeps[&MR] = std::move(InitSectionSymbols);
  }

  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::registerInitSections(
    jitlink::LinkGraph &G, JITDylib &JD, bool IsBootstraping) {
  SmallVector<ExecutorAddrRange> ELFNixPlatformSecs;
  SmallVector<jitlink::Section *> InitSections;

  LLVM_DEBUG(dbgs() << "ELFNixPlatform::registerInitSections\n");

  for (auto &Sec : G.sections()) {
    if (isELFInitializerSection(Sec.getName())) {
      InitSections.push_back(&Sec);
      jitlink::SectionRange R(Sec);
      ELFNixPlatformSecs.push_back(R.getRange());
    }
  }

  // Dump the scraped inits.
  LLVM_DEBUG({
    dbgs() << "ELFNixPlatform: Scraped " << G.getName() << " init sections:\n";
    for (auto *Sec : InitSections) {
      jitlink::SectionRange R(*Sec);
      dbgs() << "  " << Sec->getName() << ": " << R.getRange() << "\n";
    }
  });
  using SPSRegisterInitSectionsArgs =
      SPSArgList<SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>>;

  ExecutorAddr HeaderAddr;
  {
    std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
    auto I = MP.JITDylibToHandleAddr.find(&JD);
    assert(I != MP.JITDylibToHandleAddr.end() && "No header registered for JD");
    assert(I->second && "Null header registered for JD");
    HeaderAddr = I->second;
  }

  if (IsBootstraping) {
    auto &JBS = MP.JDBootstrapStates[&JD];
    for (auto &I : ELFNixPlatformSecs)
      JBS.Initializers.push_back(I);
    // G.allocActions().push_back(
    //    {{},
    //     cantFail(
    //       WrapperFunctionCall::Create<SPSRegisterInitSectionsArgs>(
    //           MP.orc_rt_elfnix_deregister_init_sections, HeaderAddr,
    //           ELFNixPlatformSecs))});
  } else {
    G.allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<SPSRegisterInitSectionsArgs>(
             MP.orc_rt_elfnix_register_init_sections, HeaderAddr,
             ELFNixPlatformSecs)),
         cantFail(WrapperFunctionCall::Create<SPSRegisterInitSectionsArgs>(
             MP.orc_rt_elfnix_deregister_init_sections, HeaderAddr,
             ELFNixPlatformSecs))});
  }

  return Error::success();
}

Error ELFNixPlatform::ELFNixPlatformPlugin::fixTLVSectionsAndEdges(
    jitlink::LinkGraph &G, JITDylib &JD) {

  for (auto *Sym : G.external_symbols()) {
    if (Sym->getName() == "__tls_get_addr") {
      Sym->setName("___orc_rt_elfnix_tls_get_addr");
    } else if (Sym->getName() == "__tlsdesc_resolver") {
      Sym->setName("___orc_rt_elfnix_tlsdesc_resolver");
    }
  }

  auto *TLSInfoEntrySection = G.findSectionByName("$__TLSINFO");

  if (TLSInfoEntrySection) {
    std::optional<uint64_t> Key;
    {
      std::lock_guard<std::mutex> Lock(MP.PlatformMutex);
      auto I = MP.JITDylibToPThreadKey.find(&JD);
      if (I != MP.JITDylibToPThreadKey.end())
        Key = I->second;
    }
    if (!Key) {
      if (auto KeyOrErr = MP.createPThreadKey())
        Key = *KeyOrErr;
      else
        return KeyOrErr.takeError();
    }

    uint64_t PlatformKeyBits =
        support::endian::byte_swap(*Key, G.getEndianness());

    for (auto *B : TLSInfoEntrySection->blocks()) {
      // FIXME: The TLS descriptor byte length may different with different
      // ISA
      assert(B->getSize() == (G.getPointerSize() * 2) &&
             "TLS descriptor must be 2 words length");
      auto TLSInfoEntryContent = B->getMutableContent(G);
      memcpy(TLSInfoEntryContent.data(), &PlatformKeyBits, G.getPointerSize());
    }
  }

  return Error::success();
}

} // End namespace orc.
} // End namespace llvm.
