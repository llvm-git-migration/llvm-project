// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_TIME_ZONE_TYPES_H
#define _LIBCPP___CHRONO_TIME_ZONE_TYPES_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#  include <__chrono/day.h>
#  include <__chrono/duration.h>
#  include <__chrono/month.h>
#  include <__chrono/weekday.h>
#  include <__chrono/year.h>
#  include <__config>
#  include <string>
#  include <variant>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono::__tz {

// Sun>=8   first Sunday on or after the eighth
// Sun<=25  last Sunday on or before the 25th
struct _LIBCPP_AVAILABILITY_TZDB __constrained_weekday {
  /*  year_month_day operator()(year __year, month __month);*/ // needed but not implemented

  weekday __weekday;
  enum __comparison_t { __le, __ge } __comparison;
  day __day;
};

// The on field has a few alternative presentations
//  5        the fifth of the month
//  lastSun  the last Sunday in the month
//  lastMon  the last Monday in the month
//  Sun>=8   first Sunday on or after the eighth
//  Sun<=25  last Sunday on or before the 25th
using __on = variant<day, weekday_last, __constrained_weekday>;

enum class _LIBCPP_AVAILABILITY_TZDB __clock { __local, __standard, __universal };

struct _LIBCPP_AVAILABILITY_TZDB __at {
  seconds __time{0};
  __tz::__clock __clock{__tz::__clock::__local};
};

struct _LIBCPP_AVAILABILITY_TZDB __save {
  seconds __time;
  bool __is_dst;
};

// The names of the fields match the fields of a Rule.
struct _LIBCPP_AVAILABILITY_TZDB __rule {
  year __from;
  year __to;
  month __in_month; // __in is a reserved name
  __tz::__on __on;
  __tz::__at __at;
  __tz::__save __save;
  string __letters;
};

struct _LIBCPP_AVAILABILITY_TZDB __continuation {
  seconds __stdoff;

  // The RULES is either a SAVE or a NAME.
  // The size_t is used as cache. After loading the rules they are
  // sorted and remain stable, then an index in the vector can be
  // used.
  // If this field contains - then standard time always
  // applies. This is indicated by the monostate.
  using __rules_t = variant<monostate, __tz::__save, string, size_t>;

  __rules_t __rules;

  string __format;
  // TODO TZDB until can be contain more than just a year.
  // Parts of the UNTIL, the optional parts are default initialized
  //    optional<year> __until_;
  year __year = chrono::year::min();
  month __in_month{January}; // __in is a reserved name
  __tz::__on __on{chrono::day{1}};
  __tz::__at __at{chrono::seconds{0}, __tz::__clock::__local};
};

} // namespace chrono::__tz

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#endif // _LIBCPP___CHRONO_TIME_ZONE_TYPES_H
