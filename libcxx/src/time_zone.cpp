//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#include <chrono>

#include "include/tzdb/time_zone.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI time_zone::time_zone(unique_ptr<time_zone::__impl>&& __p)
    : __impl_(std::move(__p)) {
  _LIBCPP_ASSERT_NON_NULL(__impl_ != nullptr, "initialized time_zone without a valid pimpl object");
}

_LIBCPP_EXPORTED_FROM_ABI time_zone::~time_zone() = default;

_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI string_view time_zone::name() const noexcept { return __impl_->name(); }

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD
