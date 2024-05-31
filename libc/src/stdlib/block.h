//===-- Implementation header for a block of memory -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_BLOCK_H
#define LLVM_LIBC_SRC_STDLIB_BLOCK_H

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

namespace internal {
// Types of corrupted blocks, and functions to crash with an error message
// corresponding to each type.
enum BlockStatus {
  kValid,
  kMisaligned,
  kPrevMismatched,
  kNextMismatched,
};
} // namespace internal

/// Returns the value rounded down to the nearest multiple of alignment.
constexpr size_t AlignDown(size_t value, size_t alignment) {
  __builtin_mul_overflow(value / alignment, alignment, &value);
  return value;
}

/// Returns the value rounded down to the nearest multiple of alignment.
template <typename T> constexpr T *AlignDown(T *value, size_t alignment) {
  return reinterpret_cast<T *>(
      AlignDown(reinterpret_cast<size_t>(value), alignment));
}

/// Returns the value rounded up to the nearest multiple of alignment.
constexpr size_t AlignUp(size_t value, size_t alignment) {
  __builtin_add_overflow(value, alignment - 1, &value);
  return AlignDown(value, alignment);
}

/// Returns the value rounded up to the nearest multiple of alignment.
template <typename T> constexpr T *AlignUp(T *value, size_t alignment) {
  return reinterpret_cast<T *>(
      AlignUp(reinterpret_cast<size_t>(value), alignment));
}

using ByteSpan = cpp::span<LIBC_NAMESPACE::cpp::byte>;
using cpp::optional;

/// Memory region with links to adjacent blocks.
///
/// The blocks do not encode their size directly. Instead, they encode offsets
/// to the next and previous blocks using the type given by the `OffsetType`
/// template parameter. The encoded offsets are simply the offsets divded by the
/// minimum block alignment, `kAlignment`.
///
/// The `kAlignment` constant provided by the derived block is typically the
/// minimum value of `alignof(OffsetType)`. Since the addressable range of a
/// block is given by `std::numeric_limits<OffsetType>::max() *
/// kAlignment`, it may be advantageous to set a higher alignment if it allows
/// using a smaller offset type, even if this wastes some bytes in order to
/// align block headers.
///
/// Blocks will always be aligned to a `kAlignment` boundary. Block sizes will
/// always be rounded up to a multiple of `kAlignment`.
///
/// As an example, the diagram below represents two contiguous
/// `Block<uint32_t, true, 8>`s. The indices indicate byte offsets:
///
/// @code{.unparsed}
/// Block 1:
/// +---------------------+------+--------------+
/// | Header              | Info | Usable space |
/// +----------+----------+------+--------------+
/// | Prev     | Next     |      |              |
/// | 0......3 | 4......7 | 8..9 | 10.......280 |
/// | 00000000 | 00000046 | 8008 |  <app data>  |
/// +----------+----------+------+--------------+
/// Block 2:
/// +---------------------+------+--------------+
/// | Header              | Info | Usable space |
/// +----------+----------+------+--------------+
/// | Prev     | Next     |      |              |
/// | 0......3 | 4......7 | 8..9 | 10......1056 |
/// | 00000046 | 00000106 | 6008 | f7f7....f7f7 |
/// +----------+----------+------+--------------+
/// @endcode
///
/// The overall size of the block (e.g. 280 bytes) is given by its next offset
/// multiplied by the alignment (e.g. 0x106 * 4). Also, the next offset of a
/// block matches the previous offset of its next block. The first block in a
/// list is denoted by having a previous offset of `0`.
///
/// @tparam   OffsetType  Unsigned integral type used to encode offsets. Larger
///                       types can address more memory, but consume greater
///                       overhead.
/// @tparam   kAlign      Sets the overall alignment for blocks. Minimum is
///                       `alignof(OffsetType)` (the default). Larger values can
///                       address more memory, but consume greater overhead.
template <typename OffsetType = uintptr_t, size_t kAlign = alignof(OffsetType)>
class Block {
public:
  using offset_type = OffsetType;
  static_assert(cpp::is_unsigned_v<offset_type>,
                "offset type must be unsigned");

  static constexpr size_t kAlignment = cpp::max(kAlign, alignof(offset_type));
  static constexpr size_t kBlockOverhead = AlignUp(sizeof(Block), kAlignment);

  // No copy or move.
  Block(const Block &other) = delete;
  Block &operator=(const Block &other) = delete;

  /// @brief Creates the first block for a given memory region.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: Returns a block representing the region.
  ///
  ///    INVALID_ARGUMENT: The region is null.
  ///
  ///    RESOURCE_EXHAUSTED: The region is too small for a block.
  ///
  ///    OUT_OF_RANGE: The region is too big to be addressed using
  ///    ``OffsetType``.
  ///
  /// @endrst
  static optional<Block *> Init(ByteSpan region);

  /// @returns  A pointer to a `Block`, given a pointer to the start of the
  ///           usable space inside the block.
  ///
  /// This is the inverse of `UsableSpace()`.
  ///
  /// @warning  This method does not do any checking; passing a random
  ///           pointer will return a non-null pointer.
  static Block *FromUsableSpace(void *usable_space) {
    auto *bytes = reinterpret_cast<cpp::byte *>(usable_space);
    return reinterpret_cast<Block *>(bytes - kBlockOverhead);
  }

  /// @returns The total size of the block in bytes, including the header.
  size_t OuterSize() const { return next_ * kAlignment; }

  /// @returns The number of usable bytes inside the block.
  size_t InnerSize() const { return OuterSize() - kBlockOverhead; }

  /// @returns The number of bytes requested using AllocFirst, AllocLast, or
  /// Resize.
  size_t RequestedSize() const { return InnerSize() - padding_; }

  /// @returns A pointer to the usable space inside this block.
  cpp::byte *UsableSpace() {
    return reinterpret_cast<cpp::byte *>(this) + kBlockOverhead;
  }

  /// Marks the block as free and merges it with any free neighbors.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer. If neither member is free, the returned pointer will point to the
  /// original block. Otherwise, it will point to the new, larger block created
  /// by merging adjacent free blocks together.
  static void Free(Block *&block);

  /// Attempts to split this block.
  ///
  /// If successful, the block will have an inner size of `new_inner_size`,
  /// rounded up to a `kAlignment` boundary. The remaining space will be
  /// returned as a new block.
  ///
  /// This method may fail if the remaining space is too small to hold a new
  /// block. If this method fails for any reason, the original block is
  /// unmodified.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer with a pointer to the new, smaller block.
  ///
  /// TODO(b/326509341): Remove from the public API when FreeList is no longer
  /// in use.
  ///
  /// @pre The block must not be in use.
  ///
  /// @returns @rst
  ///
  /// .. pw-status-codes::
  ///
  ///    OK: The split completed successfully.
  ///
  ///    FAILED_PRECONDITION: This block is in use and cannot be split.
  ///
  ///    OUT_OF_RANGE: The requested size for this block is greater
  ///    than the current ``inner_size``.
  ///
  ///    RESOURCE_EXHAUSTED: The remaining space is too small to hold a
  ///    new block.
  ///
  /// @endrst
  static optional<Block *> Split(Block *&block, size_t new_inner_size);

  /// Merges this block with the one that comes after it.
  ///
  /// This method is static in order to consume and replace the given block
  /// pointer with a pointer to the new, larger block.
  static bool MergeNext(Block *&block);

  /// Fetches the block immediately after this one.
  ///
  /// For performance, this always returns a block pointer, even if the returned
  /// pointer is invalid. The pointer is valid if and only if `Last()` is false.
  ///
  /// Typically, after calling `Init` callers may save a pointer past the end of
  /// the list using `Next()`. This makes it easy to subsequently iterate over
  /// the list:
  /// @code{.cpp}
  ///   auto result = Block<>::Init(byte_span);
  ///   Block<>* begin = *result;
  ///   Block<>* end = begin->Next();
  ///   ...
  ///   for (auto* block = begin; block != end; block = block->Next()) {
  ///     // Do something which each block.
  ///   }
  /// @endcode
  Block *Next() const;

  /// @copydoc `Next`.
  static Block *NextBlock(const Block *block) {
    return block == nullptr ? nullptr : block->Next();
  }

  /// @returns The block immediately before this one, or a null pointer if this
  /// is the first block.
  Block *Prev() const;

  /// @copydoc `Prev`.
  static Block *PrevBlock(const Block *block) {
    return block == nullptr ? nullptr : block->Prev();
  }

  /// Returns the current alignment of a block.
  size_t Alignment() const { return Used() ? info_.alignment : 1; }

  /// Indicates whether the block is in use.
  ///
  /// @returns `true` if the block is in use or `false` if not.
  bool Used() const { return info_.used; }

  /// Indicates whether this block is the last block or not (i.e. whether
  /// `Next()` points to a valid block or not). This is needed because
  /// `Next()` points to the end of this block, whether there is a valid
  /// block there or not.
  ///
  /// @returns `true` is this is the last block or `false` if not.
  bool Last() const { return info_.last; }

  /// Marks this block as in use.
  void MarkUsed() { info_.used = 1; }

  /// Marks this block as free.
  void MarkFree() { info_.used = 0; }

  /// Marks this block as the last one in the chain.
  void MarkLast() { info_.last = 1; }

  /// Clears the last bit from this block.
  void ClearLast() { info_.last = 1; }

  /// @brief Checks if a block is valid.
  ///
  /// @returns `true` if and only if the following conditions are met:
  /// * The block is aligned.
  /// * The prev/next fields match with the previous and next blocks.
  /// * The poisoned bytes are not damaged (if poisoning is enabled).
  bool IsValid() const { return CheckStatus() == internal::kValid; }

private:
  /// Consumes the block and returns as a span of bytes.
  static ByteSpan AsBytes(Block *&&block);

  /// Consumes the span of bytes and uses it to construct and return a block.
  static Block *AsBlock(size_t prev_outer_size, ByteSpan bytes);

  Block(size_t prev_outer_size, size_t outer_size);

  /// Returns a `BlockStatus` that is either kValid or indicates the reason why
  /// the block is invalid.
  ///
  /// If the block is invalid at multiple points, this function will only return
  /// one of the reasons.
  internal::BlockStatus CheckStatus() const;

  /// Like `Split`, but assumes the caller has already checked to parameters to
  /// ensure the split will succeed.
  static Block *SplitImpl(Block *&block, size_t new_inner_size);

  /// Offset (in increments of the minimum alignment) from this block to the
  /// previous block. 0 if this is the first block.
  offset_type prev_ = 0;

  /// Offset (in increments of the minimum alignment) from this block to the
  /// next block. Valid even if this is the last block, since it equals the
  /// size of the block.
  offset_type next_ = 0;

  /// Information about the current state of the block:
  /// * If the `used` flag is set, the block's usable memory has been allocated
  ///   and is being used.
  /// * If the `poisoned` flag is set and the `used` flag is clear, the block's
  ///   usable memory contains a poison pattern that will be checked when the
  ///   block is allocated.
  /// * If the `last` flag is set, the block does not have a next block.
  /// * If the `used` flag is set, the alignment represents the requested value
  ///   when the memory was allocated, which may be less strict than the actual
  ///   alignment.
  struct {
    uint16_t used : 1;
    uint16_t poisoned : 1;
    uint16_t last : 1;
    uint16_t alignment : 13;
  } info_;

  /// Number of bytes allocated beyond what was requested. This will be at most
  /// the minimum alignment, i.e. `alignof(offset_type).`
  uint16_t padding_ = 0;
} __attribute__((packed, aligned(kAlign)));

// Public template method implementations.

inline ByteSpan GetAlignedSubspan(ByteSpan bytes, size_t alignment) {
  if (bytes.data() == nullptr) {
    return ByteSpan();
  }
  auto unaligned_start = reinterpret_cast<uintptr_t>(bytes.data());
  auto aligned_start = AlignUp(unaligned_start, alignment);
  auto unaligned_end = unaligned_start + bytes.size();
  auto aligned_end = AlignDown(unaligned_end, alignment);
  if (aligned_end <= aligned_start) {
    return ByteSpan();
  }
  return bytes.subspan(aligned_start - unaligned_start,
                       aligned_end - aligned_start);
}

template <typename OffsetType, size_t kAlign>
optional<Block<OffsetType, kAlign> *>
Block<OffsetType, kAlign>::Init(ByteSpan region) {
  optional<ByteSpan> result = GetAlignedSubspan(region, kAlignment);
  if (!result) {
    return {};
  }
  region = result.value();
  if (region.size() < kBlockOverhead) {
    return {};
  }
  if (cpp::numeric_limits<OffsetType>::max() < region.size() / kAlignment) {
    return {};
  }
  Block *block = AsBlock(0, region);
  block->MarkLast();
  return block;
}

template <typename OffsetType, size_t kAlign>
void Block<OffsetType, kAlign>::Free(Block *&block) {
  if (block == nullptr) {
    return;
  }
  block->MarkFree();
  Block *prev = block->Prev();
  if (MergeNext(prev)) {
    block = prev;
  }
  MergeNext(block);
}

template <typename OffsetType, size_t kAlign>
optional<Block<OffsetType, kAlign> *>
Block<OffsetType, kAlign>::Split(Block *&block, size_t new_inner_size) {
  if (block == nullptr) {
    return {};
  }
  if (block->Used()) {
    return {};
  }
  size_t old_inner_size = block->InnerSize();
  new_inner_size = AlignUp(new_inner_size, kAlignment);
  if (old_inner_size < new_inner_size) {
    return {};
  }
  if (old_inner_size - new_inner_size < kBlockOverhead) {
    return {};
  }
  return SplitImpl(block, new_inner_size);
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *
Block<OffsetType, kAlign>::SplitImpl(Block *&block, size_t new_inner_size) {
  size_t prev_outer_size = block->prev_ * kAlignment;
  size_t outer_size1 = new_inner_size + kBlockOverhead;
  bool is_last = block->Last();
  ByteSpan bytes = AsBytes(cpp::move(block));
  Block *block1 = AsBlock(prev_outer_size, bytes.subspan(0, outer_size1));
  Block *block2 = AsBlock(outer_size1, bytes.subspan(outer_size1));
  if (is_last) {
    block2->MarkLast();
  } else {
    block2->Next()->prev_ = block2->next_;
  }
  block = cpp::move(block1);
  return block2;
}

template <typename OffsetType, size_t kAlign>
bool Block<OffsetType, kAlign>::MergeNext(Block *&block) {
  if (block == nullptr) {
    return false;
  }
  if (block->Last()) {
    return false;
  }
  Block *next = block->Next();
  if (block->Used() || next->Used()) {
    return false;
  }
  size_t prev_outer_size = block->prev_ * kAlignment;
  bool is_last = next->Last();
  ByteSpan prev_bytes = AsBytes(cpp::move(block));
  ByteSpan next_bytes = AsBytes(cpp::move(next));
  size_t outer_size = prev_bytes.size() + next_bytes.size();
  cpp::byte *merged = ::new (prev_bytes.data()) cpp::byte[outer_size];
  block = AsBlock(prev_outer_size, ByteSpan(merged, outer_size));
  if (is_last) {
    block->MarkLast();
  } else {
    block->Next()->prev_ = block->next_;
  }
  return true;
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::Next() const {
  uintptr_t addr = Last() ? 0 : reinterpret_cast<uintptr_t>(this) + OuterSize();
  return reinterpret_cast<Block *>(addr);
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *Block<OffsetType, kAlign>::Prev() const {
  uintptr_t addr =
      (prev_ == 0) ? 0
                   : reinterpret_cast<uintptr_t>(this) - (prev_ * kAlignment);
  return reinterpret_cast<Block *>(addr);
}

// Private template method implementations.

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign>::Block(size_t prev_outer_size, size_t outer_size) {
  prev_ = prev_outer_size / kAlignment;
  next_ = outer_size / kAlignment;
  info_.used = 0;
  info_.poisoned = 0;
  info_.last = 0;
  info_.alignment = kAlignment;
}

template <typename OffsetType, size_t kAlign>
ByteSpan Block<OffsetType, kAlign>::AsBytes(Block *&&block) {
  size_t block_size = block->OuterSize();
  cpp::byte *bytes = new (cpp::move(block)) cpp::byte[block_size];
  return {bytes, block_size};
}

template <typename OffsetType, size_t kAlign>
Block<OffsetType, kAlign> *
Block<OffsetType, kAlign>::AsBlock(size_t prev_outer_size, ByteSpan bytes) {
  return ::new (bytes.data()) Block(prev_outer_size, bytes.size());
}

template <typename OffsetType, size_t kAlign>
internal::BlockStatus Block<OffsetType, kAlign>::CheckStatus() const {
  if (reinterpret_cast<uintptr_t>(this) % kAlignment != 0) {
    return internal::kMisaligned;
  }
  if (!Last() && (this >= Next() || this != Next()->Prev())) {
    return internal::kNextMismatched;
  }
  if (Prev() && (this <= Prev() || this != Prev()->Next())) {
    return internal::kPrevMismatched;
  }
  return internal::kValid;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_BLOCK_H
