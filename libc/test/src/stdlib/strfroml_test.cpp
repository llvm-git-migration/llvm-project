//===-- Unittests for strfroml --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfroml.h"
#include "test/UnitTest/Test.h"

#define ASSERT_STREQ_LEN_STRFROML(actual_written, actual_str, expected_str)    \
  EXPECT_EQ(actual_written, static_cast<int>(sizeof(expected_str) - 1));       \
  EXPECT_STREQ(actual_str, expected_str);

TEST(LlvmLibcStrfromlTest, FloatDecimalFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfroml(buff, 40, "%f", 1.0L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1.000000");

  written = LIBC_NAMESPACE::strfroml(buff, 10, "%.f", -2.5L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "-2");
}

TEST(LlvmLibcStrfromlTest, FloatExpFormat) {
  char buff[100];
  int written;
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  written = LIBC_NAMESPACE::strfroml(buff, 90, "%.9e", 1000000000500000000.1L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1.000000001e+18");

  written = LIBC_NAMESPACE::strfroml(buff, 90, "%.9e", 1000000000500000000.0L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1.000000000e+18");

  written =
      LIBC_NAMESPACE::strfroml(buff, 90, "%e", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1.189731e+4932");
#endif
}

TEST(LlvmLibcStrfromlTest, FloatDecimalAutoFormat) {
  char buff[100];
  int written;

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  written =
      LIBC_NAMESPACE::strfroml(buff, 99, "%g", 0xf.fffffffffffffffp+16380L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1.18973e+4932");

  written = LIBC_NAMESPACE::strfroml(buff, 99, "%g", 0xa.aaaaaaaaaaaaaabp-7L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0.0833333");

  written = LIBC_NAMESPACE::strfroml(buff, 99, "%g", 9.99999999999e-100L);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "1e-99");
#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
}

TEST(LlvmLibcStrfromlTest, FloatHexExpFormat) {
  char buff[100];
  int written;

  written = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0xc.ccccccccccccccdp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x1.999999999999ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN_STRFROML(written, buff,
                            "0x1.999999999999999999999999999ap-4");
#endif

  written = LIBC_NAMESPACE::strfroml(buff, 20, "%.1a", 0.1L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0xc.dp-7");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x1.ap-4");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x1.ap-4");
#endif

  written = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 1.0e1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0xf.38db1f9dd3dac05p+3318");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN_STRFROML(written, buff,
                            "0x1.e71b63f3ba7b580af1a52d2a7379p+3321");
#endif

  written = LIBC_NAMESPACE::strfroml(buff, 50, "%a", 1.0e-1000L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x8.68a9188a89e1467p-3325");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x0p+0");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN_STRFROML(written, buff,
                            "0x1.0d152311513c28ce202627c06ec2p-3322");
#endif

  written =
      LIBC_NAMESPACE::strfroml(buff, 50, "%.1a", 0xf.fffffffffffffffp16380L);
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x1.0p+16384");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "inf");
#elif defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128)
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0x2.0p+16383");
#endif
}

TEST(LlvmLibcStrfromfTest, ImproperFormatString) {
  char buff[100];
  int written;
  written = LIBC_NAMESPACE::strfroml(
      buff, 37, "A simple string with no conversions.", 1.0);
  ASSERT_STREQ_LEN_STRFROML(written, buff,
                            "A simple string with no conversions.");

  written = LIBC_NAMESPACE::strfroml(
      buff, 37, "%A simple string with one conversion, should overwrite.", 1.0);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "0X8P-3");

  written =
      LIBC_NAMESPACE::strfroml(buff, 74,
                               "A simple string with one conversion in %A "
                               "between, writes string as it is",
                               1.0);
  ASSERT_STREQ_LEN_STRFROML(
      written, buff,
      "A simple string with one conversion in %A between, "
      "writes string as it is");

  written = LIBC_NAMESPACE::strfroml(
      buff, 36, "A simple string with one conversion", 1.0);
  ASSERT_STREQ_LEN_STRFROML(written, buff,
                            "A simple string with one conversion");

  written = LIBC_NAMESPACE::strfroml(buff, 20, "%1f", 1234567890.0);
  ASSERT_STREQ_LEN_STRFROML(written, buff, "%1f");
}

TEST(LlvmLibcStrfromfTest, InsufficientBufsize) {
  char buff[20];
  int written;

  written = LIBC_NAMESPACE::strfroml(buff, 5, "%f", 1234567890.0);
  EXPECT_EQ(written, 17);
  ASSERT_STREQ(buff, "1234");

  written = LIBC_NAMESPACE::strfroml(buff, 5, "%.5f", 1.05);
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "1.05");

  written = LIBC_NAMESPACE::strfroml(buff, 0, "%g", 1.0);
  EXPECT_EQ(written, 1);
}
