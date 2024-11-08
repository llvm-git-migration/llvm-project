

#ifndef LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H
#define LLVM_CLANG_AST_INTERP_BITCAST_BUFFER_H

#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <sstream>

enum class Endian { Little, Big };

static inline bool bitof(std::byte B, unsigned BitIndex) {
  assert(BitIndex < 8);
  return (B & (std::byte{1} << BitIndex)) != std::byte{0};
}

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

  /// \p Data must be in the given endianness.
  void pushData(const std::byte *data, size_t BitOffset, size_t BitWidth,
                Endian DataEndianness) {
    for (unsigned It = 0; It != BitWidth; ++It) {
      bool BitValue;
      BitValue = bitof(data[It / 8], It % 8);
      if (!BitValue)
        continue;

      unsigned DstBit;
      if (DataEndianness == Endian::Little)
        DstBit = BitOffset + It;
      else
        DstBit = size() - BitOffset - BitWidth + It;

      unsigned DstByte = (DstBit / 8);
      Data[DstByte] |= std::byte{1} << (DstBit % 8);
    }
  }

  std::unique_ptr<std::byte[]> copyBits(unsigned BitOffset, unsigned BitWidth,
                                        unsigned FullBitWidth,
                                        Endian DataEndianness) const {
    assert(BitWidth <= FullBitWidth);
    assert(fullByte(FullBitWidth));
    auto Out = std::make_unique<std::byte[]>(FullBitWidth / 8);

    for (unsigned It = 0; It != BitWidth; ++It) {
      unsigned BitIndex;
      if (DataEndianness == Endian::Little)
        BitIndex = BitOffset + It;
      else
        BitIndex = size() - BitWidth - BitOffset + It;

      bool BitValue = bitof(Data[BitIndex / 8], BitIndex % 8);
      if (!BitValue)
        continue;
      unsigned DstBit = It;
      unsigned DstByte = (DstBit / 8);
      Out[DstByte] |= std::byte{1} << (DstBit % 8);
    }

    return Out;
  }

#if 0
  template<typename T>
  static std::string hex(T t) {
    std::stringstream stream;
    stream << std::hex << (int)t;
    return std::string(stream.str());
  }


  void dump(bool AsHex = true) const {
    llvm::errs() << "LSB\n  ";
    unsigned LineLength = 0;
    for (unsigned I = 0; I != (FinalBitSize / 8); ++I) {
      std::byte B = Data[I];
      if (AsHex) {
        std::stringstream stream;
        stream << std::hex << (int)B;
        llvm::errs() << stream.str();
        LineLength += stream.str().size() + 1;
      } else {
        llvm::errs() << std::bitset<8>((int)B).to_string();
        LineLength += 8 + 1;
        // llvm::errs() << (int)B;
      }
      llvm::errs() << ' ';
    }
    llvm::errs() << '\n';

    for (unsigned I = 0; I != LineLength; ++I)
      llvm::errs() << ' ';
    llvm::errs() << "MSB\n";
  }
#endif
};

#endif
