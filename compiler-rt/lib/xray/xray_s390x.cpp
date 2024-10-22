//===-- xray_s390x.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of s390x routines.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_interface_internal.h"
#include <cassert>
#include <cstring>

namespace __xray {

bool patchFunctionEntry(const bool Enable, uint32_t FuncId,
                        const XRaySledEntry &Sled,
                        void (*Trampoline)()) XRAY_NEVER_INSTRUMENT {
  const uint64_t Address = Sled.address();
  if (Enable) {
    // The resulting code is:
    //   stmg    %r2, %r15, 16(%r15)
    //   llilf   %2, FuncID
    //   brasl   %r14, __xray_FunctionEntry@GOT
    // The FuncId and the stmg instruction must be written.

    // Write FuncId into llilf.
    reinterpret_cast<uint32_t *>(Address)[2] = FuncId;
    // Write last part of stmg.
    reinterpret_cast<uint16_t *>(Address)[2] = 0x24;
    // Write first part of stmg.
    reinterpret_cast<uint32_t *>(Address)[0] = 0xeb2ff010;
  } else {
    // j +16 instructions.
    *reinterpret_cast<uint32_t *>(Address) = 0xa7f4000b;
  }
  return true;
}

bool patchFunctionExit(const bool Enable, uint32_t FuncId,
                       const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  const uint64_t Address = Sled.address();
  if (Enable) {
    // The resulting code is:
    //   stmg    %r2, %r15, 24(%r15)
    //   llilf   %2,FuncID
    //   j       __xray_FunctionEntry@GOT
    // The FuncId and the stmg instruction must be written.

    // Write FuncId into llilf.
    reinterpret_cast<uint32_t *>(Address)[2] = FuncId;
    // Write last part of of stmg.
    reinterpret_cast<uint16_t *>(Address)[2] = 0x24;
    // Write first part of stmg.
    reinterpret_cast<uint32_t *>(Address)[0] = 0xeb2ff010;
  } else {
    // br %14 instruction.
    *reinterpret_cast<uint16_t *>(Address) = 0x07fe;
  }
  return true;
}

bool patchFunctionTailExit(const bool Enable, const uint32_t FuncId,
                           const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  return patchFunctionExit(Enable, FuncId, Sled);
}

bool patchCustomEvent(const bool Enable, const uint32_t FuncId,
                      const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // FIXME: Implement.
  return false;
}

bool patchTypedEvent(const bool Enable, const uint32_t FuncId,
                     const XRaySledEntry &Sled) XRAY_NEVER_INSTRUMENT {
  // FIXME: Implement.
  return false;
}

} // namespace __xray

extern "C" void __xray_ArgLoggerEntry() XRAY_NEVER_INSTRUMENT {
  // FIXME: this will have to be implemented in the trampoline assembly file.
}
