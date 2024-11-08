//===--------------------- BitcastBuffer.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H
#define LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H

#include <cassert>
#include <cstddef>
#include <memory>

namespace clang {
namespace interp {

enum class Endian { Little, Big };

/// Returns the value of the bit in the given sequence of bytes.
static inline bool bitof(const std::byte *B, unsigned BitIndex) {
  return (B[BitIndex / 8] & (std::byte{1} << (BitIndex % 8))) != std::byte{0};
}

/// Returns whether \p N is a full byte offset or size.
static inline bool fullByte(unsigned N) { return N % 8 == 0; }

/// Track what bits have been initialized to known values and which ones
/// have indeterminate value.
/// All offsets are in bits.
struct BitcastBuffer {
  size_t FinalBitSize = 0;
  std::unique_ptr<std::byte[]> Data;

  BitcastBuffer(size_t FinalBitSize) : FinalBitSize(FinalBitSize) {
    assert(fullByte(FinalBitSize));
    unsigned ByteSize = FinalBitSize / 8;
    Data = std::make_unique<std::byte[]>(ByteSize);
  }

  size_t size() const { return FinalBitSize; }

  bool allInitialized() const {
    // FIXME: Implement.
    return true;
  }

  void pushData(const std::byte *data, size_t BitOffset, size_t BitWidth,
                Endian DataEndianness);
  std::unique_ptr<std::byte[]> copyBits(unsigned BitOffset, unsigned BitWidth,
                                        unsigned FullBitWidth,
                                        Endian DataEndianness) const;
};

} // namespace interp
} // namespace clang
#endif
