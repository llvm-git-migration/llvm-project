//===-- SBAddressRange.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBAddressRange.h"
#include "Utils.h"
#include "lldb/API/SBAddress.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Utility/Instrumentation.h"
#include <cstddef>
#include <memory>

using namespace lldb;
using namespace lldb_private;

SBAddressRange::SBAddressRange()
    : m_opaque_up(std::make_unique<AddressRange>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBAddressRange::SBAddressRange(const SBAddressRange &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBAddressRange::SBAddressRange(lldb::addr_t file_addr, lldb::addr_t byte_size)
    : m_opaque_up(std::make_unique<AddressRange>(file_addr, byte_size)) {
  LLDB_INSTRUMENT_VA(this, file_addr, byte_size);
}

SBAddressRange::~SBAddressRange() = default;

const SBAddressRange &SBAddressRange::operator=(const SBAddressRange &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

void SBAddressRange::Clear() {
  LLDB_INSTRUMENT_VA(this);

  m_opaque_up.reset();
}

bool SBAddressRange::IsValid() const {
  return m_opaque_up && m_opaque_up->IsValid();
}

lldb::SBAddress SBAddressRange::GetBaseAddress() const {
  LLDB_INSTRUMENT_VA(this);

  assert(m_opaque_up.get() && "AddressRange is NULL");
  return lldb::SBAddress(m_opaque_up->GetBaseAddress());
}

lldb::addr_t SBAddressRange::GetByteSize() const {
  LLDB_INSTRUMENT_VA(this);

  assert(m_opaque_up.get() && "AddressRange is NULL");
  return m_opaque_up->GetByteSize();
}

AddressRange &SBAddressRange::ref() {
  assert(m_opaque_up.get() && "AddressRange is NULL");
  return *m_opaque_up;
}

const AddressRange &SBAddressRange::ref() const {
  assert(m_opaque_up.get() && "AddressRange is NULL");
  return *m_opaque_up;
}
