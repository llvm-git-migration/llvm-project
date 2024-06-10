// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_ZONED_TIME_H
#define _LIBCPP___CHRONO_ZONED_TIME_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#  include <__chrono/duration.h>
#  include <__chrono/system_clock.h>
#  include <__chrono/time_zone.h>
#  include <__chrono/tzdb_list.h>
#  include <__config>
#  include <__fwd/string_view.h>
#  include <__type_traits/common_type.h>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono {

template <class>
struct zoned_traits {};

template <>
struct zoned_traits<const time_zone*> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static const time_zone* default_zone() { return chrono::locate_zone("UTC"); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static const time_zone* locate_zone(string_view __name) {
    return chrono::locate_zone(__name);
  }
};

template <class _Duration, class _TimeZonePtr = const time_zone*>
class zoned_time {
  // [time.zone.zonedtime.ctor]/2
  static_assert(__is_duration<_Duration>::value,
                "the program is ill-formed since _Duration is not a specialization of std::chrono::duration");

  using __traits = zoned_traits<_TimeZonePtr>;

public:
  using duration = common_type_t<_Duration, seconds>;

  _LIBCPP_HIDE_FROM_ABI zoned_time()
    requires requires { __traits::default_zone(); }
      : __zone_{__traits::default_zone()}, __tp_{} {}

  _LIBCPP_HIDE_FROM_ABI zoned_time(const zoned_time&)            = default;
  _LIBCPP_HIDE_FROM_ABI zoned_time& operator=(const zoned_time&) = default;

  _LIBCPP_HIDE_FROM_ABI zoned_time(const sys_time<_Duration>& __tp)
    requires requires { __traits::default_zone(); }
      : __zone_{__traits::default_zone()}, __tp_{__tp} {}

  _LIBCPP_HIDE_FROM_ABI explicit zoned_time(_TimeZonePtr __zone) : __zone_{std::move(__zone)}, __tp_{} {}

  _LIBCPP_HIDE_FROM_ABI explicit zoned_time(string_view __name)
    requires(requires { __traits::locate_zone(string_view{}); } &&
             // constructible_from<zoned_time, decltype(__traits::locate_zone(string_view{}))>
             // would create a dependency on itself. Instead depend on the fact
             // a constructor taking a _TimeZonePtr exists.
             constructible_from<_TimeZonePtr, decltype(__traits::locate_zone(string_view{}))>)
      : __zone_{__traits::locate_zone(__name)}, __tp_{} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _TimeZonePtr get_time_zone() const { return __zone_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI sys_time<duration> get_sys_time() const { return __tp_; }

private:
  _TimeZonePtr __zone_;
  sys_time<duration> __tp_;
};

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#endif // _LIBCPP___CHRONO_ZONED_TIME_H
