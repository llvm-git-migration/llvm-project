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

// Tests the IANA database zones parsing and operations.
// This is not part of the public tzdb interface.

#include <cassert>
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

static const std::chrono::tzdb& parse(std::string_view input) {
  write(input);
  return std::chrono::reload_tzdb();
}

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
  test_exception("Z", "corrupt tzdb: expected whitespace");

  test_exception("Z ", "corrupt tzdb: expected a string");

  test_exception("Z n", "corrupt tzdb: expected whitespace");

  test_exception("Z n ", "corrupt tzdb: expected a digit");
  test_exception("Z n x", "corrupt tzdb: expected a digit");
  test_exception("Z n +", "corrupt tzdb: expected a digit");

  test_exception("Z n 0", "corrupt tzdb: expected whitespace");

  test_exception("Z n 0 ", "corrupt tzdb: expected a string");

  test_exception("Z n 0 r", "corrupt tzdb: expected whitespace");

  test_exception("Z n 0 r ", "corrupt tzdb: expected a string");
}

static void test_name() {
  const std::chrono::tzdb& result = parse(
      R"(
Z n 0 r f
)");
  assert(result.zones.size() == 1);
  assert(result.zones[0].name() == "n");
}

static void test_stdoff() {
  const std::chrono::tzdb& result = parse(
      R"(
# Based on the examples in the man page.
# Note the input is not expected to have fractional seconds, they are truncated.
Zo na 2 r f
Zon nb 2:00 r f
Zone nc 01:28:14 r f
zONE nd 00:19:32.10 r f
zoNe ne 12:00 r f
Z nf 15:00 r f
Z ng 24:00 r f
Z nh 260:00 r f
Z ni -2:30 r f
Z nj - r f
)");

  assert(result.zones.size() == 10);
  for (std::size_t i = 0; i < result.zones.size(); ++i)
    assert(result.zones[0].__continuations().size() == 1);

  assert(result.zones[0].__continuations()[0].__stdoff == std::chrono::hours(2));
  assert(result.zones[1].__continuations()[0].__stdoff == std::chrono::hours(2));
  assert(result.zones[2].__continuations()[0].__stdoff ==
         std::chrono::hours(1) + std::chrono::minutes(28) + std::chrono::seconds(14));
  assert(result.zones[3].__continuations()[0].__stdoff == std::chrono::minutes(19) + std::chrono::seconds(32));
  assert(result.zones[4].__continuations()[0].__stdoff == std::chrono::hours(12));
  assert(result.zones[5].__continuations()[0].__stdoff == std::chrono::hours(15));
  assert(result.zones[6].__continuations()[0].__stdoff == std::chrono::hours(24));
  assert(result.zones[7].__continuations()[0].__stdoff == std::chrono::hours(260));
  assert(result.zones[8].__continuations()[0].__stdoff == -(std::chrono::hours(2) + std::chrono::minutes(30)));
  assert(result.zones[9].__continuations()[0].__stdoff == std::chrono::hours(0)); // The man page expresses it in hours
}

static void test_rules() {
  const std::chrono::tzdb& result = parse(
      R"(
Z na 0 - f
Z nb 0 r f
Z nc 0 2d f
Z nd 0 2:00s f
Z ne 0 0 f
Z nf 0 0:00:01 f
Z ng 0 -0:00:01 f
)");

  assert(result.zones.size() == 7);
  for (std::size_t i = 0; i < result.zones.size(); ++i)
    assert(result.zones[0].__continuations().size() == 1);

  assert(std::holds_alternative<std::monostate>(result.zones[0].__continuations()[0].__rules));
  assert(std::get<std::string>(result.zones[1].__continuations()[0].__rules) == "r");

  assert(std::get<std::chrono::__tz::__save>(result.zones[2].__continuations()[0].__rules).__time ==
         std::chrono::hours(2));
  assert(std::get<std::chrono::__tz::__save>(result.zones[2].__continuations()[0].__rules).__is_dst == true);

  assert(std::get<std::chrono::__tz::__save>(result.zones[3].__continuations()[0].__rules).__time ==
         std::chrono::hours(2));
  assert(std::get<std::chrono::__tz::__save>(result.zones[3].__continuations()[0].__rules).__is_dst == false);

  assert(std::get<std::chrono::__tz::__save>(result.zones[4].__continuations()[0].__rules).__time ==
         std::chrono::hours(0));
  assert(std::get<std::chrono::__tz::__save>(result.zones[4].__continuations()[0].__rules).__is_dst == false);

  assert(std::get<std::chrono::__tz::__save>(result.zones[5].__continuations()[0].__rules).__time ==
         std::chrono::seconds(1));
  assert(std::get<std::chrono::__tz::__save>(result.zones[5].__continuations()[0].__rules).__is_dst == true);

  assert(std::get<std::chrono::__tz::__save>(result.zones[6].__continuations()[0].__rules).__time ==
         -std::chrono::seconds(1));
  assert(std::get<std::chrono::__tz::__save>(result.zones[6].__continuations()[0].__rules).__is_dst == true);
}

static void test_format() {
  const std::chrono::tzdb& result = parse(
      R"(
Z n 0 r f
)");
  assert(result.zones.size() == 1);
  assert(result.zones[0].__continuations().size() == 1);
  assert(result.zones[0].__continuations()[0].__format == "f");
}

static void test_until() {
  const std::chrono::tzdb& result = parse(
      R"(
Z na 0 r f
Z nb 0 r f 1000
Z nc 0 r f -1000 N
Z nd 0 r f ma S 31
Z ne 0 r f 0 jA LASTw
Z nf 0 r f -42 jUN m<=1
Z ng 0 r f 42 jul Su>=12
Z nh 0 r f 42 JUl 1 2w
Z ni 0 r f 42 July 1 01:28:14u
Z nj 0 r f 42 Jul 1 -
)");
  assert(result.zones.size() == 10);
  for (std::size_t i = 0; i < result.zones.size(); ++i)
    assert(result.zones[0].__continuations().size() == 1);

  std::chrono::__tz::__constrained_weekday r;

  assert(result.zones[0].__continuations()[0].__year == std::chrono::year::min());
  assert(result.zones[0].__continuations()[0].__in_month == std::chrono::January);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[1].__continuations()[0].__year == std::chrono::year(1000));
  assert(result.zones[1].__continuations()[0].__in_month == std::chrono::January);
  assert(std::get<std::chrono::day>(result.zones[1].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[1].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[1].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[2].__continuations()[0].__year == std::chrono::year(-1000));
  assert(result.zones[2].__continuations()[0].__in_month == std::chrono::November);
  assert(std::get<std::chrono::day>(result.zones[2].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[2].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[2].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[3].__continuations()[0].__year == std::chrono::year::max());
  assert(result.zones[3].__continuations()[0].__in_month == std::chrono::September);
  assert(std::get<std::chrono::day>(result.zones[3].__continuations()[0].__on) == std::chrono::day(31));
  assert(result.zones[3].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[3].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[4].__continuations()[0].__year == std::chrono::year(0));
  assert(result.zones[4].__continuations()[0].__in_month == std::chrono::January);
  assert(std::get<std::chrono::weekday_last>(result.zones[4].__continuations()[0].__on) ==
         std::chrono::weekday_last{std::chrono::Wednesday});
  assert(result.zones[4].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[4].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[5].__continuations()[0].__year == std::chrono::year(-42));
  assert(result.zones[5].__continuations()[0].__in_month == std::chrono::June);
  r = std::get<std::chrono::__tz::__constrained_weekday>(result.zones[5].__continuations()[0].__on);
  assert(r.__weekday == std::chrono::Monday);
  assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__le);
  assert(r.__day == std::chrono::day(1));
  assert(result.zones[5].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[5].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[6].__continuations()[0].__year == std::chrono::year(42));
  assert(result.zones[6].__continuations()[0].__in_month == std::chrono::July);
  r = std::get<std::chrono::__tz::__constrained_weekday>(result.zones[6].__continuations()[0].__on);
  assert(r.__weekday == std::chrono::Sunday);
  assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
  assert(r.__day == std::chrono::day(12));
  assert(result.zones[6].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[6].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[7].__continuations()[0].__year == std::chrono::year(42));
  assert(result.zones[7].__continuations()[0].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[7].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[7].__continuations()[0].__at.__time == std::chrono::hours(2));
  assert(result.zones[7].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[8].__continuations()[0].__year == std::chrono::year(42));
  assert(result.zones[8].__continuations()[0].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[8].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[8].__continuations()[0].__at.__time ==
         std::chrono::hours(1) + std::chrono::minutes(28) + std::chrono::seconds(14));
  assert(result.zones[8].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__universal);

  assert(result.zones[9].__continuations()[0].__year == std::chrono::year(42));
  assert(result.zones[9].__continuations()[0].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[9].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[9].__continuations()[0].__at.__time ==
         std::chrono::hours(0)); // The man page expresses it in hours
  assert(result.zones[9].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);
}

static void test_continuation() {
  const std::chrono::tzdb& result = parse(
      R"(
Z na 0 r f
0 r f 1000
0 r f -1000 N
0 r f ma S 31
0 r f 0 Ja lastW
0 r f -42 Jun M<=1
0 r f 42 Jul Su>=12
0 r f 42 Jul 1 2w
0 r f 42 Jul 1 01:28:14u
0 r f 42 Jul 1 -
)");

  assert(result.zones.size() == 1);
  assert(result.zones[0].__continuations().size() == 10);

  std::chrono::__tz::__constrained_weekday r;

  assert(result.zones[0].__continuations()[0].__year == std::chrono::year::min());
  assert(result.zones[0].__continuations()[0].__in_month == std::chrono::January);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[0].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[0].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[0].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[1].__year == std::chrono::year(1000));
  assert(result.zones[0].__continuations()[1].__in_month == std::chrono::January);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[1].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[1].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[1].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[2].__year == std::chrono::year(-1000));
  assert(result.zones[0].__continuations()[2].__in_month == std::chrono::November);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[2].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[2].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[2].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[3].__year == std::chrono::year::max());
  assert(result.zones[0].__continuations()[3].__in_month == std::chrono::September);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[3].__on) == std::chrono::day(31));
  assert(result.zones[0].__continuations()[3].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[3].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[4].__year == std::chrono::year(0));
  assert(result.zones[0].__continuations()[4].__in_month == std::chrono::January);
  assert(std::get<std::chrono::weekday_last>(result.zones[0].__continuations()[4].__on) ==
         std::chrono::weekday_last{std::chrono::Wednesday});
  assert(result.zones[0].__continuations()[4].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[4].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[5].__year == std::chrono::year(-42));
  assert(result.zones[0].__continuations()[5].__in_month == std::chrono::June);
  r = std::get<std::chrono::__tz::__constrained_weekday>(result.zones[0].__continuations()[5].__on);
  assert(r.__weekday == std::chrono::Monday);
  assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__le);
  assert(r.__day == std::chrono::day(1));
  assert(result.zones[0].__continuations()[5].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[5].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[6].__year == std::chrono::year(42));
  assert(result.zones[0].__continuations()[6].__in_month == std::chrono::July);
  r = std::get<std::chrono::__tz::__constrained_weekday>(result.zones[0].__continuations()[6].__on);
  assert(r.__weekday == std::chrono::Sunday);
  assert(r.__comparison == std::chrono::__tz::__constrained_weekday::__ge);
  assert(r.__day == std::chrono::day(12));
  assert(result.zones[0].__continuations()[6].__at.__time == std::chrono::seconds(0));
  assert(result.zones[0].__continuations()[6].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[7].__year == std::chrono::year(42));
  assert(result.zones[0].__continuations()[7].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[7].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[7].__at.__time == std::chrono::hours(2));
  assert(result.zones[0].__continuations()[7].__at.__clock == std::chrono::__tz::__clock::__local);

  assert(result.zones[0].__continuations()[8].__year == std::chrono::year(42));
  assert(result.zones[0].__continuations()[8].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[8].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[8].__at.__time ==
         std::chrono::hours(1) + std::chrono::minutes(28) + std::chrono::seconds(14));
  assert(result.zones[0].__continuations()[8].__at.__clock == std::chrono::__tz::__clock::__universal);

  assert(result.zones[0].__continuations()[9].__year == std::chrono::year(42));
  assert(result.zones[0].__continuations()[9].__in_month == std::chrono::July);
  assert(std::get<std::chrono::day>(result.zones[0].__continuations()[9].__on) == std::chrono::day(1));
  assert(result.zones[0].__continuations()[9].__at.__time ==
         std::chrono::hours(0)); // The man page expresses it in hours
  assert(result.zones[0].__continuations()[9].__at.__clock == std::chrono::__tz::__clock::__local);
}

int main(int, const char**) {
  test_invalid();
  test_name();
  test_stdoff();
  test_rules();
  test_format();
  test_until();

  test_continuation();

  return 0;
}
