// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_TIME_ZONE_H
#define _LIBCPP___CHRONO_TIME_ZONE_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#  include <__chrono/time_zone_types.h>
#  include <__compare/strong_order.h>
#  include <__config>
#  include <string>
#  include <string_view>
#  include <vector>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono {

class _LIBCPP_AVAILABILITY_TZDB time_zone {
public:
  explicit _LIBCPP_HIDE_FROM_ABI time_zone(string&& __name) : __name_(std::move(__name)) {}

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI string_view name() const noexcept { return __name_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI vector<__tz::__continuation>& __continuations() { return __continuations_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const vector<__tz::__continuation>& __continuations() const {
    return __continuations_;
  }

private:
  string __name_;
  // Note the first line has a name + __continuation, the other lines
  // are just __continuations. So there is always at least one item in
  // the vector.
  vector<__tz::__continuation> __continuations_;
};

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_HIDE_FROM_ABI inline bool
operator==(const time_zone& __x, const time_zone& __y) noexcept {
  return __x.name() == __y.name();
}

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_HIDE_FROM_ABI inline strong_ordering
operator<=>(const time_zone& __x, const time_zone& __y) noexcept {
  return __x.name() <=> __y.name();
}

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#endif // _LIBCPP___CHRONO_TIME_ZONE_H
