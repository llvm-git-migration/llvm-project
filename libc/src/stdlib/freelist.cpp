#include "freelist.h"
#include "src/__support/CPP/span.h"
#include <stddef.h>

using LIBC_NAMESPACE::cpp::span;

namespace pw::allocator {

template <size_t kNumBuckets>
Status FreeList<kNumBuckets>::AddChunk(span<LIBC_NAMESPACE::cpp::byte> chunk) {
  // Check that the size is enough to actually store what we need
  if (chunk.size() < sizeof(FreeListNode)) {
    return Status::OutOfRange();
  }

  union {
    FreeListNode *node;
    LIBC_NAMESPACE::cpp::byte *bytes;
  } aliased;

  aliased.bytes = chunk.data();

  unsigned short chunk_ptr = FindChunkPtrForSize(chunk.size(), false);

  // Add it to the correct list.
  aliased.node->size = chunk.size();
  aliased.node->next = chunks_[chunk_ptr];
  chunks_[chunk_ptr] = aliased.node;

  return OkStatus();
}

template <size_t kNumBuckets>
span<LIBC_NAMESPACE::cpp::byte>
FreeList<kNumBuckets>::FindChunk(size_t size) const {
  if (size == 0) {
    return span<LIBC_NAMESPACE::cpp::byte>();
  }

  unsigned short chunk_ptr = FindChunkPtrForSize(size, true);

  // Check that there's data. This catches the case where we run off the
  // end of the array
  if (chunks_[chunk_ptr] == nullptr) {
    return span<LIBC_NAMESPACE::cpp::byte>();
  }

  // Now iterate up the buckets, walking each list to find a good candidate
  for (size_t i = chunk_ptr; i < chunks_.size(); i++) {
    union {
      FreeListNode *node;
      LIBC_NAMESPACE::cpp::byte *data;
    } aliased;
    aliased.node = chunks_[static_cast<unsigned short>(i)];

    while (aliased.node != nullptr) {
      if (aliased.node->size >= size) {
        return span<LIBC_NAMESPACE::cpp::byte>(aliased.data,
                                               aliased.node->size);
      }

      aliased.node = aliased.node->next;
    }
  }

  // If we get here, we've checked every block in every bucket. There's
  // nothing that can support this allocation.
  return span<LIBC_NAMESPACE::cpp::byte>();
}

template <size_t kNumBuckets>
Status
FreeList<kNumBuckets>::RemoveChunk(span<LIBC_NAMESPACE::cpp::byte> chunk) {
  unsigned short chunk_ptr = FindChunkPtrForSize(chunk.size(), true);

  // Walk that list, finding the chunk.
  union {
    FreeListNode *node;
    LIBC_NAMESPACE::cpp::byte *data;
  } aliased, aliased_next;

  // Check head first.
  if (chunks_[chunk_ptr] == nullptr) {
    return Status::NotFound();
  }

  aliased.node = chunks_[chunk_ptr];
  if (aliased.data == chunk.data()) {
    chunks_[chunk_ptr] = aliased.node->next;

    return OkStatus();
  }

  // No? Walk the nodes.
  aliased.node = chunks_[chunk_ptr];

  while (aliased.node->next != nullptr) {
    aliased_next.node = aliased.node->next;
    if (aliased_next.data == chunk.data()) {
      // Found it, remove this node out of the chain
      aliased.node->next = aliased_next.node->next;
      return OkStatus();
    }

    aliased.node = aliased.node->next;
  }

  return Status::NotFound();
}

template <size_t kNumBuckets>
unsigned short FreeList<kNumBuckets>::FindChunkPtrForSize(size_t size,
                                                          bool non_null) const {
  unsigned short chunk_ptr = 0;
  for (chunk_ptr = 0u; chunk_ptr < sizes_.size(); chunk_ptr++) {
    if (sizes_[chunk_ptr] >= size &&
        (!non_null || chunks_[chunk_ptr] != nullptr)) {
      break;
    }
  }

  return chunk_ptr;
}

} // namespace pw::allocator
