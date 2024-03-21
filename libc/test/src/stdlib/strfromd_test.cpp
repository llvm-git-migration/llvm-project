//===-- Unittests for strfromd --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfromd.h"
#include "test/UnitTest/Test.h"

// Use when the buffsize is sufficient to hold all of the result string,
// including the null byte.
#define ASSERT_STREQ_LEN_STRFROMD(actual_written, actual_str, expected_str)    \
  EXPECT_EQ(actual_written, static_cast<int>(sizeof(expected_str) - 1));       \
  EXPECT_STREQ(actual_str, expected_str);

TEST(LlvmLibcStrfromdTest, FloatDecimalFormat) {
  char buff[500];
  int written;

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%F", -1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-1.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", -1.234567);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-1.234567");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1.5);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.500000");

  written = LIBC_NAMESPACE::strfromd(buff, 499, "%f", 1e300);
  ASSERT_STREQ_LEN_STRFROMD(
      written, buff,
      "100000000000000005250476025520442024870446858110815915491585411551180245"
      "798890819578637137508044786404370444383288387817694252323536043057564479"
      "218478670698284838720092657580373783023379478809005936895323497079994508"
      "111903896764088007465274278014249457925878882005684283811566947219638686"
      "5459400540160.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.1);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.100000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1234567890123456789.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1234567890123456768.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 9999999999999.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "9999999999999.990234");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 0.1);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.100000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 1234567890123456789.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1234567890123456768.000000");

  written = LIBC_NAMESPACE::strfromd(buff, 99, "%f", 9999999999999.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "9999999999999.990234");

  // Precision Tests
  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.2f", 9999999999999.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "9999999999999.99");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.1f", 9999999999999.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "10000000000000.0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.5f", 1.25);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.25000");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.0f", 1.25);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.20f", 1.234e-10);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.00000000012340000000");
}

TEST(LlvmLibcStrfromdTest, FloatHexExpFormat) {
  char buff[101];
  int written;

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%a", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0x1p+0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%A", -1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-0X1P+0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%a", -0x1.abcdef12345p0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-0x1.abcdef12345p+0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%A", 0x1.abcdef12345p0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0X1.ABCDEF12345P+0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%a", 0.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0x0p+0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%a", 1.0e100);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0x1.249ad2594c37dp+332");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%a", 0.1);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0x1.999999999999ap-4");
}

TEST(LlvmLibcStrfromdTest, FloatDecimalExpFormat) {
  char buff[101];
  int written;

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.000000e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%E", -1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-1.000000E+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", -1.234567);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "-1.234567e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 0.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.000000e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1.5);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.500000e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1e300);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.000000e+300");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%e", 1234567890123456789.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.234568e+18");

  // Precision Tests
  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.0e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 1.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "2.0e+00");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.1e", 9.99);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.0e+01");
}

TEST(LlvmLibcStrfromdTest, FloatDecimalAutoFormat) {
  char buff[120];
  int written;

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 1234567890123456789.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.23457e+18");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 9999990000000.00);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "9.99999e+12");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 9999999000000.00);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1e+13");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 0xa.aaaaaaaaaaaaaabp-7);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.0833333");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%g", 0.00001);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1e-05");

  // Precision Tests
  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.0g", 0.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.2g", 0.1);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0.1");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.2g", 1.09);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.1");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.15g", 22.25);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "22.25");

  written = LIBC_NAMESPACE::strfromd(buff, 100, "%.20g", 1.234e-10);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "1.2340000000000000814e-10");
}

TEST(LlvmLibcStrfromdTest, ImproperFormatString) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfromd(
      buff, 37, "A simple string with no conversions.", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff,
                            "A simple string with no conversions.");

  written = LIBC_NAMESPACE::strfromd(
      buff, 37, "%A simple string with one conversion, should overwrite.", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff, "0X1P+0");

  written =
      LIBC_NAMESPACE::strfromd(buff, 74,
                               "A simple string with one conversion in %A "
                               "between, writes string as it is",
                               1.0);
  ASSERT_STREQ_LEN_STRFROMD(
      written, buff,
      "A simple string with one conversion in %A between, "
      "writes string as it is");

  written = LIBC_NAMESPACE::strfromd(
      buff, 36, "A simple string with one conversion", 1.0);
  ASSERT_STREQ_LEN_STRFROMD(written, buff,
                            "A simple string with one conversion");
}

// Test the result when the buffsize is not sufficient to hold
// the result string.
TEST(LlvmLibcStrfromdTest, InsufficientBuffsize) {
  char buff[20];
  int written;

  written = LIBC_NAMESPACE::strfromd(buff, 5, "%f", 1234567890.0);
  EXPECT_EQ(written, 17);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::strfromd(buff, 5, "%.5f", 1.05);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "1.05");

  written = LIBC_NAMESPACE::strfromd(buff, 0, "%g", 1.0);
  EXPECT_EQ(written, 1);
}
