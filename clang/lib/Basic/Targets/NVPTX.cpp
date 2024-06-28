//===--- NVPTX.cpp - Implement NVPTX target feature support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements NVPTX TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "Targets.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

static constexpr Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::HEADER, ALL_LANGUAGES},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, FEATURE, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#include "clang/Basic/BuiltinsNVPTX.def"
};

const char *const NVPTXTargetInfo::GCCRegNames[] = {"r0"};

NVPTXTargetInfo::NVPTXTargetInfo(const llvm::Triple &Triple,
                                 const TargetOptions &Opts,
                                 unsigned TargetPointerWidth)
    : TargetInfo(Triple) {
  assert((TargetPointerWidth == 32 || TargetPointerWidth == 64) &&
         "NVPTX only supports 32- and 64-bit modes.");

  PTXVersion = 32;
  for (const StringRef Feature : Opts.FeaturesAsWritten) {
    int PTXV;
    if (!Feature.starts_with("+ptx") ||
        Feature.drop_front(4).getAsInteger(10, PTXV))
      continue;
    PTXVersion = PTXV; // TODO: should it be max(PTXVersion, PTXV)?
  }

  TLSSupported = false;
  VLASupported = false;
  AddrSpaceMap = &NVPTXAddrSpaceMap;
  UseAddrSpaceMapMangling = true;
  // __bf16 is always available as a load/store only type.
  BFloat16Width = BFloat16Align = 16;
  BFloat16Format = &llvm::APFloat::BFloat();

  // Define available target features
  // These must be defined in sorted order!
  NoAsmVariants = true;
  GPU = GpuArch::UNUSED;

  // PTX supports f16 as a fundamental type.
  HasLegalHalfType = true;
  HasFloat16 = true;

  if (TargetPointerWidth == 32)
    resetDataLayout("e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  else if (Opts.NVPTXUseShortPointers)
    resetDataLayout(
        "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  else
    resetDataLayout("e-i64:64-i128:128-v16:16-v32:32-n16:32:64");

  // If possible, get a TargetInfo for our host triple, so we can match its
  // types.
  llvm::Triple HostTriple(Opts.HostTriple);
  if (!HostTriple.isNVPTX())
    HostTarget = AllocateTarget(llvm::Triple(Opts.HostTriple), Opts);

  // If no host target, make some guesses about the data layout and return.
  if (!HostTarget) {
    LongWidth = LongAlign = TargetPointerWidth;
    PointerWidth = PointerAlign = TargetPointerWidth;
    switch (TargetPointerWidth) {
    case 32:
      SizeType = TargetInfo::UnsignedInt;
      PtrDiffType = TargetInfo::SignedInt;
      IntPtrType = TargetInfo::SignedInt;
      break;
    case 64:
      SizeType = TargetInfo::UnsignedLong;
      PtrDiffType = TargetInfo::SignedLong;
      IntPtrType = TargetInfo::SignedLong;
      break;
    default:
      llvm_unreachable("TargetPointerWidth must be 32 or 64");
    }

    MaxAtomicInlineWidth = TargetPointerWidth;
    return;
  }

  // Copy properties from host target.
  PointerWidth = HostTarget->getPointerWidth(LangAS::Default);
  PointerAlign = HostTarget->getPointerAlign(LangAS::Default);
  BoolWidth = HostTarget->getBoolWidth();
  BoolAlign = HostTarget->getBoolAlign();
  IntWidth = HostTarget->getIntWidth();
  IntAlign = HostTarget->getIntAlign();
  HalfWidth = HostTarget->getHalfWidth();
  HalfAlign = HostTarget->getHalfAlign();
  FloatWidth = HostTarget->getFloatWidth();
  FloatAlign = HostTarget->getFloatAlign();
  DoubleWidth = HostTarget->getDoubleWidth();
  DoubleAlign = HostTarget->getDoubleAlign();
  LongWidth = HostTarget->getLongWidth();
  LongAlign = HostTarget->getLongAlign();
  LongLongWidth = HostTarget->getLongLongWidth();
  LongLongAlign = HostTarget->getLongLongAlign();
  MinGlobalAlign = HostTarget->getMinGlobalAlign(/* TypeSize = */ 0,
                                                 /* HasNonWeakDef = */ true);
  NewAlign = HostTarget->getNewAlign();
  DefaultAlignForAttributeAligned =
      HostTarget->getDefaultAlignForAttributeAligned();
  SizeType = HostTarget->getSizeType();
  IntMaxType = HostTarget->getIntMaxType();
  PtrDiffType = HostTarget->getPtrDiffType(LangAS::Default);
  IntPtrType = HostTarget->getIntPtrType();
  WCharType = HostTarget->getWCharType();
  WIntType = HostTarget->getWIntType();
  Char16Type = HostTarget->getChar16Type();
  Char32Type = HostTarget->getChar32Type();
  Int64Type = HostTarget->getInt64Type();
  SigAtomicType = HostTarget->getSigAtomicType();
  ProcessIDType = HostTarget->getProcessIDType();

  UseBitFieldTypeAlignment = HostTarget->useBitFieldTypeAlignment();
  UseZeroLengthBitfieldAlignment = HostTarget->useZeroLengthBitfieldAlignment();
  UseExplicitBitFieldAlignment = HostTarget->useExplicitBitFieldAlignment();
  ZeroLengthBitfieldBoundary = HostTarget->getZeroLengthBitfieldBoundary();

  // This is a bit of a lie, but it controls __GCC_ATOMIC_XXX_LOCK_FREE, and
  // we need those macros to be identical on host and device, because (among
  // other things) they affect which standard library classes are defined, and
  // we need all classes to be defined on both the host and device.
  MaxAtomicInlineWidth = HostTarget->getMaxAtomicInlineWidth();

  // Properties intentionally not copied from host:
  // - LargeArrayMinWidth, LargeArrayAlign: Not visible across the
  //   host/device boundary.
  // - SuitableAlign: Not visible across the host/device boundary, and may
  //   correctly be different on host/device, e.g. if host has wider vector
  //   types than device.
  // - LongDoubleWidth, LongDoubleAlign: nvptx's long double type is the same
  //   as its double type, but that's not necessarily true on the host.
  //   TODO: nvcc emits a warning when using long double on device; we should
  //   do the same.
}

ArrayRef<const char *> NVPTXTargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

bool NVPTXTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Cases("ptx", "nvptx", true)
      .Default(false);
}

void NVPTXTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__PTX__");
  Builder.defineMacro("__NVPTX__");

  // Skip setting architecture dependent macros if undefined.
  if (GPU == GpuArch::UNUSED && !HostTarget)
    return;

  if (Opts.CUDAIsDevice || Opts.OpenMPIsTargetDevice || !HostTarget) {
    // Set __CUDA_ARCH__ for the GPU specified.
    std::string CUDAArchCode = [this] {
      switch (GPU) {
      case GpuArch::GFX600:
      case GpuArch::GFX601:
      case GpuArch::GFX602:
      case GpuArch::GFX700:
      case GpuArch::GFX701:
      case GpuArch::GFX702:
      case GpuArch::GFX703:
      case GpuArch::GFX704:
      case GpuArch::GFX705:
      case GpuArch::GFX801:
      case GpuArch::GFX802:
      case GpuArch::GFX803:
      case GpuArch::GFX805:
      case GpuArch::GFX810:
      case GpuArch::GFX9_GENERIC:
      case GpuArch::GFX900:
      case GpuArch::GFX902:
      case GpuArch::GFX904:
      case GpuArch::GFX906:
      case GpuArch::GFX908:
      case GpuArch::GFX909:
      case GpuArch::GFX90a:
      case GpuArch::GFX90c:
      case GpuArch::GFX940:
      case GpuArch::GFX941:
      case GpuArch::GFX942:
      case GpuArch::GFX10_1_GENERIC:
      case GpuArch::GFX1010:
      case GpuArch::GFX1011:
      case GpuArch::GFX1012:
      case GpuArch::GFX1013:
      case GpuArch::GFX10_3_GENERIC:
      case GpuArch::GFX1030:
      case GpuArch::GFX1031:
      case GpuArch::GFX1032:
      case GpuArch::GFX1033:
      case GpuArch::GFX1034:
      case GpuArch::GFX1035:
      case GpuArch::GFX1036:
      case GpuArch::GFX11_GENERIC:
      case GpuArch::GFX1100:
      case GpuArch::GFX1101:
      case GpuArch::GFX1102:
      case GpuArch::GFX1103:
      case GpuArch::GFX1150:
      case GpuArch::GFX1151:
      case GpuArch::GFX1152:
      case GpuArch::GFX12_GENERIC:
      case GpuArch::GFX1200:
      case GpuArch::GFX1201:
      case GpuArch::AMDGCNSPIRV:
      case GpuArch::Generic:
      case GpuArch::LAST:
        break;
      case GpuArch::UNKNOWN:
        assert(false && "No GPU arch when compiling CUDA device code.");
        return "";
      case GpuArch::UNUSED:
      case GpuArch::SM_20:
        return "200";
      case GpuArch::SM_21:
        return "210";
      case GpuArch::SM_30:
        return "300";
      case GpuArch::SM_32_:
        return "320";
      case GpuArch::SM_35:
        return "350";
      case GpuArch::SM_37:
        return "370";
      case GpuArch::SM_50:
        return "500";
      case GpuArch::SM_52:
        return "520";
      case GpuArch::SM_53:
        return "530";
      case GpuArch::SM_60:
        return "600";
      case GpuArch::SM_61:
        return "610";
      case GpuArch::SM_62:
        return "620";
      case GpuArch::SM_70:
        return "700";
      case GpuArch::SM_72:
        return "720";
      case GpuArch::SM_75:
        return "750";
      case GpuArch::SM_80:
        return "800";
      case GpuArch::SM_86:
        return "860";
      case GpuArch::SM_87:
        return "870";
      case GpuArch::SM_89:
        return "890";
      case GpuArch::SM_90:
      case GpuArch::SM_90a:
        return "900";
      }
      llvm_unreachable("unhandled GpuArch");
    }();
    Builder.defineMacro("__CUDA_ARCH__", CUDAArchCode);
    if (GPU == GpuArch::SM_90a)
      Builder.defineMacro("__CUDA_ARCH_FEAT_SM90_ALL", "1");
  }
}

ArrayRef<Builtin::Info> NVPTXTargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef(BuiltinInfo,
                        clang::NVPTX::LastTSBuiltin - Builtin::FirstTSBuiltin);
}
