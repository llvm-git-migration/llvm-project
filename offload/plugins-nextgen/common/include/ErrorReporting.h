//===- ErrorReporting.h - Helper to provide nice error messages ----- c++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H

#include "PluginInterface.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

class ErrorReporter {
  /// The banner printed at the beginning of an error report.
  static constexpr auto ErrorBanner = "OFFLOAD ERROR: ";

  /// Terminal color codes
  ///
  /// TODO: determine if the terminal supports colors.
  ///@{
  static constexpr auto Green = []() { return "\033[1m\033[32m"; };
  static constexpr auto Blue = []() { return "\033[1m\033[34m"; };
  static constexpr auto Red = []() { return "\033[1m\033[31m"; };
  static constexpr auto Magenta = []() { return "\033[1m\033[35m"; };
  static constexpr auto Cyan = []() { return "\033[1m\033[36m"; };
  static constexpr auto Default = []() { return "\033[1m\033[0m"; };
  ///@}

  /// The size of the getBuffer() buffer.
  static constexpr unsigned BufferSize = 1024;

  /// Return a buffer of size BufferSize that can be used for formatting.
  static char *getBuffer() {
    static char *Buffer = nullptr;
    if (!Buffer)
      Buffer = reinterpret_cast<char *>(malloc(BufferSize));
    return Buffer;
  }

  /// Return the device id as string, or n/a if not available.
  static std::string getDeviceIdStr(GenericDeviceTy *Device) {
    return Device ? std::to_string(Device->getDeviceId()) : "n/a";
  }

  /// Return a nice name for an TargetAllocTy.
  static std::string getAllocTyName(TargetAllocTy Kind) {
    switch (Kind) {
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      return "device memory";
    case TARGET_ALLOC_HOST:
      return "pinned host memory";
    case TARGET_ALLOC_SHARED:
      return "managed memory";
      break;
    }
    llvm_unreachable("Unknown target alloc kind");
  }

  /// Return a C string after \p Format has been instantiated with \p Args.
  template <typename... ArgsTy>
  static const char *getCString(const char *Format, ArgsTy &&...Args) {
    std::snprintf(getBuffer(), BufferSize, Format,
                  std::forward<ArgsTy>(Args)...);
    return getBuffer();
  }

  /// Print \p Format, instantiated with \p Args to stderr.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  static void print(const char *Format, ArgsTy &&...Args) {
    fprintf(stderr, Format, std::forward<ArgsTy>(Args)...);
  }

  /// Pretty print a stack trace.
  static void reportStackTrace(StringRef StackTrace) {
    if (StackTrace.empty())
      return;

    SmallVector<StringRef> Lines, Parts;
    StackTrace.split(Lines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    int Start = Lines.empty() || !Lines[0].contains("PrintStackTrace") ? 0 : 1;
    for (int I = Start, E = Lines.size(); I < E; ++I) {
      auto Line = Lines[I];
      Parts.clear();
      Line.split(Parts, " ", /*MaxSplit=*/2);
      if (Parts.size() != 3 || Parts[0].size() < 2 || Parts[0][0] != '#') {
        print("%s\n", Line.str().c_str());
        continue;
      }
      unsigned FrameIdx = std::stoi(Parts[0].drop_front(1).str());
      if (Start)
        FrameIdx -= 1;
      print("    %s%s%s%u %s%s%s %s\n", Magenta(),
            Parts[0].take_front().str().c_str(), Green(), FrameIdx, Blue(),
            Parts[1].str().c_str(), Default(), Parts[2].str().c_str());
    }

    printf("\n");
  }

  /// Report an error.
  static void reportError(const char *Message, StringRef StackTrace) {
    print("%s%s%s\n%s", Red(), ErrorBanner, Message, Default());
    reportStackTrace(StackTrace);
  }

  /// Report information about an allocation associated with \p ATI.
  static void reportAllocationInfo(AllocationTraceInfoTy *ATI) {
    if (!ATI)
      return;

    if (!ATI->DeallocationTrace.empty()) {
      print("%s%s%s\n%s", Cyan(), "Last deallocation:", Default());
      reportStackTrace(ATI->DeallocationTrace);
    }

    if (ATI->HostPtr)
      print("%sLast allocation of size %lu for host pointer %p:\n%s", Cyan(),
            ATI->Size, ATI->HostPtr, Default());
    else
      print("%sLast allocation of size %lu:\n%s", Cyan(), ATI->Size, Default());
    reportStackTrace(ATI->AllocationTrace);
    if (!ATI->LastAllocationInfo)
      return;

    unsigned I = 0;
    print("%sPrior allocations with the same base pointer:", Cyan());
    while (ATI->LastAllocationInfo) {
      print("\n%s", Default());
      ATI = ATI->LastAllocationInfo;
      print("%s #%u Prior deallocation of size %lu:\n%s", Cyan(), I, ATI->Size,
            Default());
      reportStackTrace(ATI->DeallocationTrace);
      if (ATI->HostPtr)
        print("%s #%u Prior allocation for host pointer %p:\n%s", Cyan(), I,
              ATI->HostPtr, Default());
      else
        print("%s #%u Prior allocation:\n%s", Cyan(), I, Default());
      reportStackTrace(ATI->AllocationTrace);
      ++I;
    }
  }

public:
  /// Check if the deallocation of \p DevicePtr is valid given \p ATI. Stores \p
  /// StackTrace to \p ATI->DeallocationTrace if there was no error.
  static void checkDeallocation(GenericDeviceTy *Device, void *DevicePtr,
                                TargetAllocTy Kind, AllocationTraceInfoTy *ATI,
                                std::string &StackTrace) {
#define DEALLOCATION_ERROR(Format, ...)                                        \
  reportError(getCString(Format, __VA_ARGS__), StackTrace);                    \
  reportAllocationInfo(ATI);                                                   \
  abort();

    if (!ATI) {
      DEALLOCATION_ERROR("deallocation of non-allocated %s: %p",
                         getAllocTyName(Kind).c_str(), DevicePtr);
    }

    if (!ATI->DeallocationTrace.empty()) {
      DEALLOCATION_ERROR("double-free of %s: %p", getAllocTyName(Kind).c_str(),
                         DevicePtr);
    }

    if (ATI->Kind != Kind) {
      DEALLOCATION_ERROR("deallocation requires %s but allocation was %s: %p",
                         getAllocTyName(Kind).c_str(),
                         getAllocTyName(ATI->Kind).c_str(), DevicePtr);
    }

    ATI->DeallocationTrace = StackTrace;

#undef DEALLOCATION_ERROR
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
