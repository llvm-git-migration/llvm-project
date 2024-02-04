//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// Tests the IANA database rules parsing and operations.
// This is not part of the public tzdb interface.

// TODO TZDB Enable the disabled parts of the tests once we can test them using the public interface.

#include <chrono>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>

#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "test_tzdb.h"

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path file                 = env.create_file("zoneinfo/tzdata.zi");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

void write(std::string_view input) {
  static int version = 0;

  std::ofstream f{file};
  f << "# version " << version++ << '\n';
  f.write(input.data(), input.size());
}
#if 0
static const std::chrono::tzdb& parse(std::string_view input) {
  write(input);
  return std::chrono::reload_tzdb();
}
#endif
static void test_exception(std::string_view input, [[maybe_unused]] std::string_view what) {
  write(input);

  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::chrono::reload_tzdb());
}

static void test_invalid() {
  test_exception("R", "corrupt tzdb: expected whitespace");

  test_exception("R ", "corrupt tzdb: expected a string");

  test_exception("R r", "corrupt tzdb: expected whitespace");

  test_exception("R r x", "corrupt tzdb: expected a digit");
  test_exception("R r +", "corrupt tzdb: expected a digit");
  test_exception("R r mx", "corrupt tzdb year: expected 'min' or 'max'");
  test_exception("R r -32768", "corrupt tzdb year: year is less than the minimum");

  test_exception("R r mix", "corrupt tzdb: expected whitespace");
  test_exception("R r 0", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 x", "corrupt tzdb: expected a digit");
  test_exception("R r 0 +", "corrupt tzdb: expected a digit");
  test_exception("R r 0 mx", "corrupt tzdb year: expected 'min' or 'max'");

  test_exception("R r 0 mix", "corrupt tzdb: expected whitespace");
  test_exception("R r 0 1", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 X", "corrupt tzdb: expected character '-'");

  test_exception("R r 0 1 -", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 - j", "corrupt tzdb month: invalid name");

  test_exception("R r 0 1 - Ja", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 - Ja +", "corrupt tzdb weekday: invalid name");
  test_exception("R r 0 1 - Ja 32", "corrupt tzdb day: value too large");
  test_exception("R r 0 1 - Ja l", "corrupt tzdb: expected string 'last'");
  test_exception("R r 0 1 - Ja last", "corrupt tzdb weekday: invalid name");
  test_exception("R r 0 1 - Ja lastS", "corrupt tzdb weekday: invalid name");
  test_exception("R r 0 1 - Ja S", "corrupt tzdb weekday: invalid name");
  test_exception("R r 0 1 - Ja Su", "corrupt tzdb on: expected '>=' or '<='");
  test_exception("R r 0 1 - Ja Su>", "corrupt tzdb: expected character '='");
  test_exception("R r 0 1 - Ja Su<", "corrupt tzdb: expected character '='");
  test_exception("R r 0 1 - Ja Su>=+", "corrupt tzdb: expected a non-zero digit");
  test_exception("R r 0 1 - Ja Su>=0", "corrupt tzdb: expected a non-zero digit");
  test_exception("R r 0 1 - Ja Su>=32", "corrupt tzdb day: value too large");

  test_exception("R r 0 1 - Ja Su>=31", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 - Ja Su>=31 ", "corrupt tzdb: expected a digit");
  test_exception("R r 0 1 - Ja Su>=31 +", "corrupt tzdb: expected a digit");

  test_exception("R r 0 1 - Ja Su>=31 1", "corrupt tzdb: expected whitespace");
  test_exception("R r 0 1 - Ja Su>=31 1a", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 - Ja Su>=31 1w 2", "corrupt tzdb: expected whitespace");
  test_exception("R r 0 1 - Ja Su>=31 1w 2a", "corrupt tzdb: expected whitespace");

  test_exception("R r 0 1 - Ja Su>=31 1w 2s", "corrupt tzdb: expected whitespace");
  test_exception("R r 0 1 - Ja Su>=31 1w 2s ", "corrupt tzdb: expected a string");
}

#if 0
static void test_name() {
  const std::chrono::tzdb& result = parse(
      R"(
R z 0 1 - Ja Su>=31 1w 2s -
rULE z 0 1 - Ja Su>=31 1w 2s -
RuLe z 0 1 - Ja Su>=31 1w 2s -
R a 0 1 - Ja Su>=31 1w 2s -
R a 0 1 - Ja Su>=31 1w 2s -
)");

  assert(result.__rules.size() == 2);
  assert(result.__rules[0].first == "a");
  assert(result.__rules[0].second.size() == 2);
  assert(result.__rules[1].first == "z");
  assert(result.__rules[1].second.size() == 3);
}

static void test_from() {
  const std::chrono::tzdb& result = parse(
      R"(
# min abbrviations
R a M 1 - Ja Su>=31 1w 2s -
R a mI 1 - Ja Su>=31 1w 2s -
R a mIN 1 - Ja Su>=31 1w 2s -

# max abbrviations
R a MA 1 - Ja Su>=31 1w 2s -
R a mAx 1 - Ja Su>=31 1w 2s -

# 40000 > max
R a -40000 1 - Ja Su>=31 1w 2s -
R a -1000 1 - Ja Su>=31 1w 2s -
R a -100 1 - Ja Su>=31 1w 2s -
R a 0000 1 - Ja Su>=31 1w 2s -
R a 100 1 - Ja Su>=31 1w 2s -
R a 1000 1 - Ja Su>=31 1w 2s -
# 40000 < min
R a 40000 1 - Ja Su>=31 1w 2s -
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 12);

  assert(result.__rules[0].second[0].__from == std::chrono::year::min());
  assert(result.__rules[0].second[1].__from == std::chrono::year::min());
  assert(result.__rules[0].second[2].__from == std::chrono::year::min());

  assert(result.__rules[0].second[3].__from == std::chrono::year::max());
  assert(result.__rules[0].second[4].__from == std::chrono::year::max());

  assert(result.__rules[0].second[5].__from == std::chrono::year::min());
  assert(result.__rules[0].second[6].__from == std::chrono::year(-1000));
  assert(result.__rules[0].second[7].__from == std::chrono::year(-100));
  assert(result.__rules[0].second[8].__from == std::chrono::year(0));
  assert(result.__rules[0].second[9].__from == std::chrono::year(100));
  assert(result.__rules[0].second[10].__from == std::chrono::year(1000));
  assert(result.__rules[0].second[11].__from == std::chrono::year::max());
}

static void test_to() {
  const std::chrono::tzdb& result = parse(
      R"(
# min abbrviations
R a 0 m - Ja Su>=31 1w 2s -
R a 0 mi - Ja Su>=31 1w 2s -
R a 0 min - Ja Su>=31 1w 2s -

# max abbrviations
R a 0 ma - Ja Su>=31 1w 2s -
R a 0 max - Ja Su>=31 1w 2s -

# 40000 > max
R a 0 -40000 - Ja Su>=31 1w 2s -
R a 0 -1000 - Ja Su>=31 1w 2s -
R a 0 -100 - Ja Su>=31 1w 2s -
R a 0 0000 - Ja Su>=31 1w 2s -
R a 0 100 - Ja Su>=31 1w 2s -
R a 0 1000 - Ja Su>=31 1w 2s -
# 40000 < min
R a 0 40000 - Ja Su>=31 1w 2s -

# only abbreviations
R a m O - Ja Su>=31 1w 2s -
R a ma oN - Ja Su>=31 1w 2s -
R a -100 onL - Ja Su>=31 1w 2s -
R a 100 oNlY - Ja Su>=31 1w 2s -
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 16);

  assert(result.__rules[0].second[0].__to == std::chrono::year::min());
  assert(result.__rules[0].second[1].__to == std::chrono::year::min());
  assert(result.__rules[0].second[2].__to == std::chrono::year::min());

  assert(result.__rules[0].second[3].__to == std::chrono::year::max());
  assert(result.__rules[0].second[4].__to == std::chrono::year::max());

  assert(result.__rules[0].second[5].__to == std::chrono::year::min());
  assert(result.__rules[0].second[6].__to == std::chrono::year(-1000));
  assert(result.__rules[0].second[7].__to == std::chrono::year(-100));
  assert(result.__rules[0].second[8].__to == std::chrono::year(0));
  assert(result.__rules[0].second[9].__to == std::chrono::year(100));
  assert(result.__rules[0].second[10].__to == std::chrono::year(1000));
  assert(result.__rules[0].second[11].__to == std::chrono::year::max());

  assert(result.__rules[0].second[12].__to == std::chrono::year::min());
  assert(result.__rules[0].second[13].__to == std::chrono::year::max());
  assert(result.__rules[0].second[14].__to == std::chrono::year(-100));
  assert(result.__rules[0].second[15].__to == std::chrono::year(100));
}

static void test_in() {
  const std::chrono::tzdb& result = parse(
      R"(
# All tests in alphabetic order to validate shortest unique abbreviation

# Shortest abbreviation valid
R s 0 1 - ap Su>=31 1w 2s -
R s 0 1 - au Su>=31 1w 2s -
R s 0 1 - d Su>=31 1w 2s -
R s 0 1 - f Su>=31 1w 2s -
R s 0 1 - ja Su>=31 1w 2s -
R s 0 1 - jul Su>=31 1w 2s -
R s 0 1 - jun Su>=31 1w 2s -
R s 0 1 - May Su>=31 1w 2s -
R s 0 1 - mar Su>=31 1w 2s -
R s 0 1 - n Su>=31 1w 2s -
R s 0 1 - o Su>=31 1w 2s -
R s 0 1 - s Su>=31 1w 2s -

# 3 letter abbreviation
R a 0 1 - APR Su>=31 1w 2s -
R a 0 1 - AUG Su>=31 1w 2s -
R a 0 1 - DEC Su>=31 1w 2s -
R a 0 1 - FEB Su>=31 1w 2s -
R a 0 1 - JAN Su>=31 1w 2s -
R a 0 1 - JUL Su>=31 1w 2s -
R a 0 1 - JUN Su>=31 1w 2s -
R a 0 1 - MAY Su>=31 1w 2s -
R a 0 1 - MAR Su>=31 1w 2s -
R a 0 1 - NOV Su>=31 1w 2s -
R a 0 1 - OCT Su>=31 1w 2s -
R a 0 1 - SEP Su>=31 1w 2s -

# Full names
R f 0 1 - ApRiL Su>=31 1w 2s -
R f 0 1 - AuGuSt Su>=31 1w 2s -
R f 0 1 - DeCeMber Su>=31 1w 2s -
R f 0 1 - FeBrUary Su>=31 1w 2s -
R f 0 1 - JaNuAry Su>=31 1w 2s -
R f 0 1 - JuLy Su>=31 1w 2s -
R f 0 1 - JuNe Su>=31 1w 2s -
R f 0 1 - MaY Su>=31 1w 2s -
R f 0 1 - MaRch Su>=31 1w 2s -
R f 0 1 - NoVemBeR Su>=31 1w 2s -
R f 0 1 - OcTobEr Su>=31 1w 2s -
R f 0 1 - SePteMbEr Su>=31 1w 2s -
)");

  assert(result.__rules.size() == 3);
  for (std::size_t i = 0; i < result.__rules.size(); ++i) {
    assert(result.__rules[i].second.size() == 12);

    assert(result.__rules[i].second[0].__in_month == std::chrono::April);
    assert(result.__rules[i].second[1].__in_month == std::chrono::August);
    assert(result.__rules[i].second[2].__in_month == std::chrono::December);
    assert(result.__rules[i].second[3].__in_month == std::chrono::February);
    assert(result.__rules[i].second[4].__in_month == std::chrono::January);
    assert(result.__rules[i].second[5].__in_month == std::chrono::July);
    assert(result.__rules[i].second[6].__in_month == std::chrono::June);
    assert(result.__rules[i].second[7].__in_month == std::chrono::May);
    assert(result.__rules[i].second[8].__in_month == std::chrono::March);
    assert(result.__rules[i].second[9].__in_month == std::chrono::November);
    assert(result.__rules[i].second[10].__in_month == std::chrono::October);
    assert(result.__rules[i].second[11].__in_month == std::chrono::September);
  }
};

static void test_on_day() {
  const std::chrono::tzdb& result = parse(
      R"(
# The parser does not validate the day as valid day of month
R a 0 1 - Fe 1 1w 2s -
R a 0 1 - Fe 10 1w 2s -
R a 0 1 - Fe 20 1w 2s -
R a 0 1 - Fe 30 1w 2s -
R a 0 1 - Fe 31 1w 2s -
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 5);
  assert(std::get<std::chrono::day>(result.__rules[0].second[0].__on) == std::chrono::day(1));
  assert(std::get<std::chrono::day>(result.__rules[0].second[1].__on) == std::chrono::day(10));
  assert(std::get<std::chrono::day>(result.__rules[0].second[2].__on) == std::chrono::day(20));
  assert(std::get<std::chrono::day>(result.__rules[0].second[3].__on) == std::chrono::day(30));
  assert(std::get<std::chrono::day>(result.__rules[0].second[4].__on) == std::chrono::day(31));
}

static void test_on_last() {
  const std::chrono::tzdb& result = parse(
      R"(
# All tests in alphabetic order to validate shortest unique abbreviation

# Shortest abbreviation valid
R s 0 1 - Ja lastF 1w 2s -
R s 0 1 - Ja lastM 1w 2s -
R s 0 1 - Ja lastSa 1w 2s -
R s 0 1 - Ja lastSu 1w 2s -
R s 0 1 - Ja lastTh 1w 2s -
R s 0 1 - Ja lastTu 1w 2s -
R s 0 1 - Ja lastW 1w 2s -

# 3 letter abbreviation
R a 0 1 - Ja lastFri 1w 2s -
R a 0 1 - Ja lastMon 1w 2s -
R a 0 1 - Ja lastSat 1w 2s -
R a 0 1 - Ja lastSun 1w 2s -
R a 0 1 - Ja lastThu 1w 2s -
R a 0 1 - Ja lastTue 1w 2s -
R a 0 1 - Ja lastWed 1w 2s -

# Full names
R f 0 1 - Ja lastFriday 1w 2s -
R f 0 1 - Ja lastMonday 1w 2s -
R f 0 1 - Ja lastSaturday 1w 2s -
R f 0 1 - Ja lastSunday 1w 2s -
R f 0 1 - Ja lastThursday 1w 2s -
R f 0 1 - Ja lastTuesday 1w 2s -
R f 0 1 - Ja lastWednesday 1w 2s -
)");

  assert(result.__rules.size() == 3);
  for (std::size_t i = 0; i < result.__rules.size(); ++i) {
    assert(result.__rules[i].second.size() == 7);

    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[0].__on) ==
           std::chrono::weekday_last(std::chrono::Friday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[1].__on) ==
           std::chrono::weekday_last(std::chrono::Monday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[2].__on) ==
           std::chrono::weekday_last(std::chrono::Saturday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[3].__on) ==
           std::chrono::weekday_last(std::chrono::Sunday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[4].__on) ==
           std::chrono::weekday_last(std::chrono::Thursday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[5].__on) ==
           std::chrono::weekday_last(std::chrono::Tuesday));
    assert(std::get<std::chrono::weekday_last>(result.__rules[i].second[6].__on) ==
           std::chrono::weekday_last(std::chrono::Wednesday));
  }
}

static void test_on_constrain() {
  const std::chrono::tzdb& result = parse(
      R"(
# Shortest abbreviation valid
R s 0 1 - Ja F>=1 1w 2s -
R s 0 1 - Ja M<=1 1w 2s -
R s 0 1 - Ja Sa>=31 1w 2s -
R s 0 1 - Ja Su<=31 1w 2s -
R s 0 1 - Ja Th>=10 1w 2s -
R s 0 1 - Ja Tu<=20 1w 2s -
R s 0 1 - Ja W>=30 1w 2s -

# 3 letter abbreviation
R a 0 1 - Ja Fri>=1 1w 2s -
R a 0 1 - Ja Mon<=1 1w 2s -
R a 0 1 - Ja Sat>=31 1w 2s -
R a 0 1 - Ja Sun<=31 1w 2s -
R a 0 1 - Ja Thu>=10 1w 2s -
R a 0 1 - Ja Tue<=20 1w 2s -
R a 0 1 - Ja Wed>=30 1w 2s -

# Full names
R f 0 1 - Ja Friday>=1 1w 2s -
R f 0 1 - Ja Monday<=1 1w 2s -
R f 0 1 - Ja Saturday>=31 1w 2s -
R f 0 1 - Ja Sunday<=31 1w 2s -
R f 0 1 - Ja Thursday>=10 1w 2s -
R f 0 1 - Ja Tuesday<=20 1w 2s -
R f 0 1 - Ja Wednesday>=30 1w 2s -

)");

  std::chrono::__tz::__constrained_weekday r;
  assert(result.__rules.size() == 3);
  for (std::size_t i = 0; i < result.__rules.size(); ++i) {
    assert(result.__rules[i].second.size() == 7);

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[0].__on);
    assert(r.__weekday == std::chrono::Friday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
    assert(r.__day == std::chrono::day(1));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[1].__on);
    assert(r.__weekday == std::chrono::Monday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__le);
    assert(r.__day == std::chrono::day(1));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[2].__on);
    assert(r.__weekday == std::chrono::Saturday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
    assert(r.__day == std::chrono::day(31));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[3].__on);
    assert(r.__weekday == std::chrono::Sunday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__le);
    assert(r.__day == std::chrono::day(31));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[4].__on);
    assert(r.__weekday == std::chrono::Thursday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
    assert(r.__day == std::chrono::day(10));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[5].__on);
    assert(r.__weekday == std::chrono::Tuesday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__le);
    assert(r.__day == std::chrono::day(20));

    r = std::get<std::chrono::__tz::__constrained_weekday>(result.__rules[i].second[6].__on);
    assert(r.__weekday == std::chrono::Wednesday);
    assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
    assert(r.__day == std::chrono::day(30));
  }
}

static void test_on() {
  test_on_day();
  test_on_last();
  test_on_constrain();
}

static void test_at() {
  const std::chrono::tzdb& result = parse(
      R"(
# Based on the examples in the man page.
# Note the input is not expected to have fractional seconds, they are truncated.
R a 0 1 - Ja Su>=31 2w 2s -
R a 0 1 - Ja Su>=31 2:00s 2s -
R a 0 1 - Ja Su>=31 01:28:14u 2s -
R a 0 1 - Ja Su>=31 00:19:32.10g 2s -
R a 0 1 - Ja Su>=31 12:00z 2s -
R a 0 1 - Ja Su>=31 15:00 2s -
R a 0 1 - Ja Su>=31 24:00 2s -
R a 0 1 - Ja Su>=31 260:00 2s -
R a 0 1 - Ja Su>=31 -2:30 2s -
R a 0 1 - Ja Su>=31 - 2s -
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 10);

  assert(result.__rules[0].second[0].__at.__time == std::chrono::hours(2));
  assert(result.__rules[0].second[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.__rules[0].second[1].__at.__time == std::chrono::hours(2));
  assert(result.__rules[0].second[1].__at.__clock == std::chrono::__tz::__clock::__standard);

  assert(result.__rules[0].second[2].__at.__time ==
         std::chrono::hours(1) + std::chrono::minutes(28) + std::chrono::seconds(14));
  assert(result.__rules[0].second[2].__at.__clock == std::chrono::__tz::__clock::__universal);

  assert(result.__rules[0].second[3].__at.__time == std::chrono::minutes(19) + std::chrono::seconds(32));
  assert(result.__rules[0].second[3].__at.__clock == std::chrono::__tz::__clock::__universal);

  assert(result.__rules[0].second[4].__at.__time == std::chrono::hours(12));
  assert(result.__rules[0].second[4].__at.__clock == std::chrono::__tz::__clock::__universal);

  assert(result.__rules[0].second[5].__at.__time == std::chrono::hours(15));
  assert(result.__rules[0].second[5].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.__rules[0].second[6].__at.__time == std::chrono::hours(24));
  assert(result.__rules[0].second[6].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.__rules[0].second[7].__at.__time == std::chrono::hours(260));
  assert(result.__rules[0].second[7].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.__rules[0].second[8].__at.__time == -(std::chrono::hours(2) + std::chrono::minutes(30)));
  assert(result.__rules[0].second[8].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.__rules[0].second[9].__at.__time == std::chrono::hours(0)); // The man page expresses it in hours
  assert(result.__rules[0].second[9].__at.__clock == std::chrono::__tz::__clock::__local);
}

static void test_save() {
  const std::chrono::tzdb& result = parse(
      R"(
R a 0 1 - Ja Su>=31 1w 2d -
R a 0 1 - Ja Su>=31 1w 2:00s -
R a 0 1 - Ja Su>=31 1w 0 -
R a 0 1 - Ja Su>=31 1w 0:00:01 -
R a 0 1 - Ja Su>=31 1w -0:00:01 -
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 5);

  assert(result.__rules[0].second[0].__save.__time == std::chrono::hours(2));
  assert(result.__rules[0].second[0].__save.__is_dst == true);

  assert(result.__rules[0].second[1].__save.__time == std::chrono::hours(2));
  assert(result.__rules[0].second[1].__save.__is_dst == false);

  assert(result.__rules[0].second[2].__save.__time == std::chrono::hours(0));
  assert(result.__rules[0].second[2].__save.__is_dst == false);

  assert(result.__rules[0].second[3].__save.__time == std::chrono::seconds(1));
  assert(result.__rules[0].second[3].__save.__is_dst == true);

  assert(result.__rules[0].second[4].__save.__time == -std::chrono::seconds(1));
  assert(result.__rules[0].second[4].__save.__is_dst == true);
}

static void test_letter() {
  const std::chrono::tzdb& result = parse(
      R"(
R a 0 1 - Ja Su>=31 1w 2s -
R a 0 1 - Ja Su>=31 1w 2s a
R a 0 1 - Ja Su>=31 1w 2s abc
)");

  assert(result.__rules.size() == 1);
  assert(result.__rules[0].second.size() == 3);

  assert(result.__rules[0].second[0].__letters == "");
  assert(result.__rules[0].second[1].__letters == "a");
  assert(result.__rules[0].second[2].__letters == "abc");
}
#endif
int main(int, const char**) {
  test_invalid();
#if 0
  test_name();
  test_from();
  test_to();
  test_in();
  test_on();
  test_at();
  test_save();
  test_letter();
#endif

  return 0;
}
