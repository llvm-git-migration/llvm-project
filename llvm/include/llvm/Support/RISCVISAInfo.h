//===-- RISCVISAInfo.h - RISC-V ISA Information -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RISCVISAINFO_H
#define LLVM_SUPPORT_RISCVISAINFO_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <map>
#include <string>
#include <vector>

namespace llvm {
void riscvExtensionsHelp(StringMap<StringRef> DescMap);

class RISCVISAInfo {
public:
  RISCVISAInfo(const RISCVISAInfo &) = delete;
  RISCVISAInfo &operator=(const RISCVISAInfo &) = delete;

  /// Represents the major and version number components of a RISC-V extension.
  struct ExtensionVersion {
    unsigned Major;
    unsigned Minor;
  };

  static bool compareExtension(const std::string &LHS, const std::string &RHS);

  /// Helper class for OrderedExtensionMap.
  struct ExtensionComparator {
    bool operator()(const std::string &LHS, const std::string &RHS) const {
      return compareExtension(LHS, RHS);
    }
  };

  /// OrderedExtensionMap is std::map, it's specialized to keep entries
  /// in canonical order of extension.
  typedef std::map<std::string, ExtensionVersion, ExtensionComparator>
      OrderedExtensionMap;

  RISCVISAInfo(unsigned XLen, OrderedExtensionMap &Exts)
      : XLen(XLen), FLen(0), MinVLen(0), MaxELen(0), MaxELenFp(0), Exts(Exts) {}

  /// Parse RISC-V ISA info from arch string.
  /// If IgnoreUnknown is set, any unrecognised extension names or
  /// extensions with unrecognised versions will be silently dropped, except
  /// for the special case of the base 'i' and 'e' extensions, where the
  /// default version will be used (as ignoring the base is not possible).
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  parseArchString(StringRef Arch, bool EnableExperimentalExtension,
                  bool ExperimentalExtensionVersionCheck = true,
                  bool IgnoreUnknown = false);

  /// Parse RISC-V ISA info from an arch string that is already in normalized
  /// form (as defined in the psABI). Unlike parseArchString, this function
  /// will not error for unrecognized extension names or extension versions.
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  parseNormalizedArchString(StringRef Arch);

  /// Parse RISC-V ISA info from feature vector.
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  parseFeatures(unsigned XLen, const std::vector<std::string> &Features);

  /// Convert RISC-V ISA info to a feature vector.
  std::vector<std::string> toFeatures(bool AddAllExtensions = false,
                                      bool IgnoreUnknown = true) const;

  const OrderedExtensionMap &getExtensions() const { return Exts; }

  unsigned getXLen() const { return XLen; }
  unsigned getFLen() const { return FLen; }
  unsigned getMinVLen() const { return MinVLen; }
  unsigned getMaxVLen() const { return 65536; }
  unsigned getMaxELen() const { return MaxELen; }
  unsigned getMaxELenFp() const { return MaxELenFp; }

  bool hasExtension(StringRef Ext) const;
  std::string toString() const;
  StringRef computeDefaultABI() const;

  static bool isSupportedExtensionFeature(StringRef Ext);
  static bool isSupportedExtension(StringRef Ext);
  static bool isSupportedExtensionWithVersion(StringRef Ext);
  static bool isSupportedExtension(StringRef Ext, unsigned MajorVersion,
                                   unsigned MinorVersion);
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  postProcessAndChecking(std::unique_ptr<RISCVISAInfo> &&ISAInfo);
  static std::string getTargetFeatureForExtension(StringRef Ext);

private:
  RISCVISAInfo(unsigned XLen)
      : XLen(XLen), FLen(0), MinVLen(0), MaxELen(0), MaxELenFp(0) {}

  unsigned XLen;
  unsigned FLen;
  unsigned MinVLen;
  unsigned MaxELen, MaxELenFp;

  OrderedExtensionMap Exts;

  void addExtension(StringRef ExtName, ExtensionVersion Version);

  Error checkDependency();

  void updateImplication();
  void updateCombination();
  void updateFLen();
  void updateMinVLen();
  void updateMaxELen();
};

namespace RISCVII {
enum VLMUL : uint8_t {
  LMUL_1 = 0,
  LMUL_2,
  LMUL_4,
  LMUL_8,
  LMUL_RESERVED,
  LMUL_F8,
  LMUL_F4,
  LMUL_F2
};

enum {
  TAIL_UNDISTURBED_MASK_UNDISTURBED = 0,
  TAIL_AGNOSTIC = 1,
  MASK_AGNOSTIC = 2,
};
} // namespace RISCVII

namespace RISCVVType {
// Is this a SEW value that can be encoded into the VTYPE format.
inline static bool isValidSEW(unsigned SEW) {
  return isPowerOf2_32(SEW) && SEW >= 8 && SEW <= 1024;
}

// Is this a LMUL value that can be encoded into the VTYPE format.
inline static bool isValidLMUL(unsigned LMUL, bool Fractional) {
  return isPowerOf2_32(LMUL) && LMUL <= 8 && (!Fractional || LMUL != 1);
}

unsigned encodeVTYPE(RISCVII::VLMUL VLMUL, unsigned SEW, bool TailAgnostic,
                     bool MaskAgnostic);

inline static RISCVII::VLMUL getVLMUL(unsigned VType) {
  unsigned VLMUL = VType & 0x7;
  return static_cast<RISCVII::VLMUL>(VLMUL);
}

// Decode VLMUL into 1,2,4,8 and fractional indicator.
std::pair<unsigned, bool> decodeVLMUL(RISCVII::VLMUL VLMUL);

inline static RISCVII::VLMUL encodeLMUL(unsigned LMUL, bool Fractional) {
  assert(isValidLMUL(LMUL, Fractional) && "Unsupported LMUL");
  unsigned LmulLog2 = Log2_32(LMUL);
  return static_cast<RISCVII::VLMUL>(Fractional ? 8 - LmulLog2 : LmulLog2);
}

inline static unsigned decodeVSEW(unsigned VSEW) {
  assert(VSEW < 8 && "Unexpected VSEW value");
  return 1 << (VSEW + 3);
}

inline static unsigned encodeSEW(unsigned SEW) {
  assert(isValidSEW(SEW) && "Unexpected SEW value");
  return Log2_32(SEW) - 3;
}

inline static unsigned getSEW(unsigned VType) {
  unsigned VSEW = (VType >> 3) & 0x7;
  return decodeVSEW(VSEW);
}

inline static bool isTailAgnostic(unsigned VType) { return VType & 0x40; }

inline static bool isMaskAgnostic(unsigned VType) { return VType & 0x80; }

void printVType(unsigned VType, raw_ostream &OS);

unsigned getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul);

std::optional<RISCVII::VLMUL>
getSameRatioLMUL(unsigned SEW, RISCVII::VLMUL VLMUL, unsigned EEW);
} // namespace RISCVVType

} // namespace llvm

#endif
