
//===-- SBAddressRangeList.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBAddressRangeList.h"
#include "Utils.h"
#include "lldb/API/SBAddressRange.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;
using namespace lldb_private;

class AddressRangeListImpl {
public:
  AddressRangeListImpl() : m_ranges() {}

  AddressRangeListImpl(const AddressRangeListImpl &rhs) = default;

  AddressRangeListImpl &operator=(const AddressRangeListImpl &rhs) {
    if (this == &rhs)
      return *this;
    m_ranges = rhs.m_ranges;
    return *this;
  }

  size_t GetSize() const { return m_ranges.size(); }

  void Reserve(size_t capacity) { return m_ranges.reserve(capacity); }

  void Append(const AddressRange &sb_region) {
    m_ranges.emplace_back(sb_region);
  }

  void Append(const AddressRangeListImpl &list) {
    Reserve(GetSize() + list.GetSize());

    for (const auto &range : list.m_ranges)
      Append(range);
  }

  void Clear() { m_ranges.clear(); }

  bool GetAddressRangeAtIndex(size_t index, AddressRange &addr_range) {
    if (index >= GetSize())
      return false;
    addr_range = m_ranges[index];
    return true;
  }

  AddressRanges &Ref() { return m_ranges; }

  const AddressRanges &Ref() const { return m_ranges; }

private:
  AddressRanges m_ranges;
};

AddressRanges &SBAddressRangeList::ref() { return m_opaque_up->Ref(); }

const AddressRanges &SBAddressRangeList::ref() const {
  return m_opaque_up->Ref();
}

SBAddressRangeList::SBAddressRangeList()
    : m_opaque_up(new AddressRangeListImpl()) {
  LLDB_INSTRUMENT_VA(this);
}

SBAddressRangeList::SBAddressRangeList(const SBAddressRangeList &rhs)
    : m_opaque_up(new AddressRangeListImpl(*rhs.m_opaque_up)) {
  LLDB_INSTRUMENT_VA(this, rhs);
}

SBAddressRangeList::~SBAddressRangeList() = default;

const SBAddressRangeList &
SBAddressRangeList::operator=(const SBAddressRangeList &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs) {
    *m_opaque_up = *rhs.m_opaque_up;
  }
  return *this;
}

uint32_t SBAddressRangeList::GetSize() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_up->GetSize();
}

bool SBAddressRangeList::GetAddressRangeAtIndex(uint64_t idx,
                                                SBAddressRange &addr_range) {
  LLDB_INSTRUMENT_VA(this, idx, addr_range);

  return m_opaque_up->GetAddressRangeAtIndex(idx, addr_range.ref());
}

void SBAddressRangeList::Clear() {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_up->Clear();
}

void SBAddressRangeList::Append(const SBAddressRange &sb_addr_range) {
  LLDB_INSTRUMENT_VA(this, sb_addr_range);

  m_opaque_up->Append(sb_addr_range.ref());
}

void SBAddressRangeList::Append(const SBAddressRangeList &sb_addr_range_list) {
  LLDB_INSTRUMENT_VA(this, sb_addr_range_list);

  m_opaque_up->Append(*sb_addr_range_list);
}

const AddressRangeListImpl *SBAddressRangeList::operator->() const {
  return m_opaque_up.get();
}

const AddressRangeListImpl &SBAddressRangeList::operator*() const {
  assert(m_opaque_up.get());
  return *m_opaque_up;
}
