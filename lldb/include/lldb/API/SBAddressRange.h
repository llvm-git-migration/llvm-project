//===-- SBAddressRange.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBADDRESSRANGE_H
#define LLDB_API_SBADDRESSRANGE_H

#include "lldb/API/SBDefines.h"

namespace lldb {

class LLDB_API SBAddressRange {
public:
  SBAddressRange();

  SBAddressRange(const lldb::SBAddressRange &rhs);

  SBAddressRange(lldb::addr_t file_addr, lldb::addr_t byte_size);

  ~SBAddressRange();

  const lldb::SBAddressRange &operator=(const lldb::SBAddressRange &rhs);

  void Clear();

  bool IsValid() const;

  /// Get the base address of the range.
  ///
  /// \return
  ///     Base address object.
  lldb::SBAddress GetBaseAddress() const;

  /// Get the byte size of this range.
  ///
  /// \return
  ///     The size in bytes of this address range.
  lldb::addr_t GetByteSize() const;

protected:
  friend class SBAddressRangeList;
  friend class SBBlock;
  friend class SBFunction;

  lldb_private::AddressRange &ref();

  const lldb_private::AddressRange &ref() const;

private:
  AddressRangeUP m_opaque_up;
};

#ifndef SWIG
bool LLDB_API operator==(const SBAddressRange &lhs, const SBAddressRange &rhs);
#endif

} // namespace lldb

#endif // LLDB_API_SBADDRESSRANGE_H
