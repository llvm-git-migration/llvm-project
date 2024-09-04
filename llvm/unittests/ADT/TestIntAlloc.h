//===- llvm/unittest/ADT/TestIntAlloc.h - Allocating integer for testing --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around an integer, that allocates on assignment.
// This class is useful when OOM-testing data containers like DenseMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_ADT_TEST_INT_ALLOC_H
#define LLVM_UNITTESTS_ADT_TEST_INT_ALLOC_H

#include <cassert>
#include <cstdint>

namespace llvm {

class IntAlloc {
  friend class IntAllocSpecial;

private:
  uint32_t m_value = 0;
  uint8_t *m_ptr = nullptr;
  bool m_allocSpecialKeys = false;

public:
  const uint32_t EMPTY = 0xFFFF;
  const uint32_t TOMBSTONE = 0xFFFE;

  IntAlloc() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  IntAlloc(uint32_t Value) { alloc(Value); }

  IntAlloc(uint32_t Value, bool AllocSpecial)
      : m_allocSpecialKeys(AllocSpecial) {
    alloc(Value);
  }

  virtual ~IntAlloc() { dealloc(); }

  IntAlloc(const IntAlloc &Other)
      : IntAlloc(Other.m_value, Other.m_allocSpecialKeys) {}

  IntAlloc(IntAlloc &&Other) noexcept {
    m_value = Other.m_value;
    Other.m_value = EMPTY;
    m_ptr = Other.m_ptr;
    Other.m_ptr = nullptr;
    m_allocSpecialKeys = Other.m_allocSpecialKeys;
  }

  void alloc(uint32_t Value) {
    // limit the number of actual allocations in mass test to reduce test
    // runtime
    bool AllocSpecial =
        m_allocSpecialKeys && (Value == TOMBSTONE || Value == EMPTY);
    if (Value <= 4 || Value % 8 == 0 || AllocSpecial) {
      m_ptr = new uint8_t(0);
    }
    m_value = Value;
  }

  void dealloc() {
    delete m_ptr;
    m_ptr = nullptr;
    m_value = EMPTY;
  }

  IntAlloc &
  operator=(const IntAlloc &other) { // NOLINT // self assignment does not occur
    assert(this != &other);
    assert(m_allocSpecialKeys == other.m_allocSpecialKeys);
    dealloc();
    alloc(other.m_value);
    return *this;
  }

  IntAlloc &operator=(
      IntAlloc &&other) noexcept { // NOLINT // self assignment does not occur
    assert(this != &other);
    assert(m_allocSpecialKeys == other.m_allocSpecialKeys);
    dealloc();
    m_value = other.m_value;
    other.m_value = 0xFFFF;
    m_ptr = other.m_ptr;
    other.m_ptr = nullptr;
    return *this;
  }

  uint32_t getValue() const { return m_value; }

  bool operator==(const IntAlloc &Other) const {
    return m_value == Other.m_value;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator uint32_t() const { return m_value; }
};

class IntAllocSpecial : public IntAlloc {
public:
  IntAllocSpecial() { m_allocSpecialKeys = true; }
  explicit IntAllocSpecial(uint32_t Value) : IntAlloc(Value, true) {}
};

} // end namespace llvm

#endif
