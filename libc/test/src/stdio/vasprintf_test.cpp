//===-- Unittests for vasprintf
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/vasprintf.h"
#include "test/UnitTest/Test.h"
#include <string>

int call_vasprintf(char **__restrict buffer, const char *__restrict format,
                   ...) {
  va_list vlist;
  va_start(vlist, format);
  int ret = LIBC_NAMESPACE::vasprintf(buffer, format, vlist);
  va_end(vlist);
  return ret;
}

TEST(LlvmLibcVASPrintfTest, SimpleNoConv) {
  char *buff = nullptr;
  int written;
  written = call_vasprintf(&buff, "A simple string with no conversions.");
  EXPECT_EQ(written, 36);
  ASSERT_STREQ(buff, "A simple string with no conversions.");
  free(buff);
}

TEST(LlvmLibcVASPrintfTest, PercentConv) {
  char *buff = nullptr;
  int written;

  written = call_vasprintf(&buff, "%%");
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "%");

  written = call_vasprintf(&buff, "abc %% def");
  EXPECT_EQ(written, 9);
  ASSERT_STREQ(buff, "abc % def");

  written = call_vasprintf(&buff, "%%%%%%");
  EXPECT_EQ(written, 3);
  ASSERT_STREQ(buff, "%%%");
  free(buff);
}

TEST(LlvmLibcVASPrintfTest, CharConv) {
  char *buff = nullptr;
  int written;

  written = call_vasprintf(&buff, "%c", 'a');
  EXPECT_EQ(written, 1);
  ASSERT_STREQ(buff, "a");

  written = call_vasprintf(&buff, "%3c %-3c", '1', '2');
  EXPECT_EQ(written, 7);
  ASSERT_STREQ(buff, "  1 2  ");

  written = call_vasprintf(&buff, "%*c", 2, '3');
  EXPECT_EQ(written, 2);
  ASSERT_STREQ(buff, " 3");
  free(buff);
}

TEST(LlvmLibcVASPrintfTest, LargeStringNoConv) {
  char *buff = nullptr;
  int written;
  auto long_str = std::string(250, 'o');
  written = call_vasprintf(&buff, long_str.c_str());
  EXPECT_EQ(written, 250);
  ASSERT_STREQ(buff, long_str.c_str());
  free(buff);
}

TEST(LlvmLibcVASPrintfTest, LargeStringCharConv) {
  char *buff = nullptr;
  int written;
  auto long_str = std::string(199, 'a');
  written = call_vasprintf(&buff, "%sbc", long_str.c_str());
  EXPECT_EQ(written, 201);
  ASSERT_STREQ(
      buff,
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabc");
  free(buff);
}