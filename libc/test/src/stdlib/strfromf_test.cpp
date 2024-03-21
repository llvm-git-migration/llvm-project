//===-- Unittests for strfromf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfromf.h"
#include "test/UnitTest/Test.h"

#define ASSERT_STREQ_LEN_STRFROMF(actual_written, actual_str, expected_str)    \
  EXPECT_EQ(actual_written, static_cast<int>(sizeof(expected_str) - 1));       \
  EXPECT_STREQ(actual_str, expected_str);

TEST(LlvmLibcStrfromfTest, DecimalFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 16, "%f", 1.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.000000");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%f", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1234567936.000000");

  written = LIBC_NAMESPACE::strfromf(buff, 67, "%.3f", 1.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.000");
}

TEST(LlvmLibcStrfromfTest, HexExpFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 0, "%a", 1234567890.0);
  EXPECT_EQ(written, 14);

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%a", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "0x1.26580cp+30");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%A", 1234567890.0);
  EXPECT_EQ(written, 14);
  ASSERT_STREQ(buff, "0X1.26580CP+30");
}

TEST(LlvmLibcStrfromfTest, DecimalExpFloatFormat) {
  char buff[100];
  int written;
  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9e", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.234567936e+09");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9E", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.234567936E+09");
}

TEST(LlvmLibcStrfromfTest, AutoDecimalFloatFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9g", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.23456794e+09");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%.9G", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "1.23456794E+09");
}

TEST(LlvmLibcStrfromfTest, ImproperFormatString) {
  char buff[100];
  int written;
  written = LIBC_NAMESPACE::strfromf(
      buff, 37, "A simple string with no conversions.", 1.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff,
                            "A simple string with no conversions.");

  written = LIBC_NAMESPACE::strfromf(
      buff, 37, "%A simple string with one conversion, should overwrite.", 1.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "0X1P+0");

  written =
      LIBC_NAMESPACE::strfromf(buff, 74,
                               "A simple string with one conversion in %A "
                               "between, writes string as it is",
                               1.0);
  ASSERT_STREQ_LEN_STRFROMF(
      written, buff,
      "A simple string with one conversion in %A between, "
      "writes string as it is");

  written = LIBC_NAMESPACE::strfromf(
      buff, 36, "A simple string with one conversion", 1.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff,
                            "A simple string with one conversion");

  written = LIBC_NAMESPACE::strfromf(buff, 20, "%1f", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROMF(written, buff, "%1f");
}

TEST(LlvmLibcStrfromfTest, InsufficientBufsize) {
  char buff[20];
  int written;

  written = LIBC_NAMESPACE::strfromf(buff, 5, "%f", 1234567890.0);
  EXPECT_EQ(written, 17);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::strfromf(buff, 5, "%.5f", 1.05);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "1.05");

  written = LIBC_NAMESPACE::strfromf(buff, 0, "%g", 1.0);
  EXPECT_EQ(written, 1);
}
