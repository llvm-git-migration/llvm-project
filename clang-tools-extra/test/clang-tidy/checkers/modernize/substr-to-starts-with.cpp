// RUN: %check_clang_tidy %s modernize-substr-to-starts-with %t -- -std=c++20 -I%clang_tidy_headers

#include <string>
#include <string_view>

void test() {
  std::string str = "hello world";
  if (str.substr(0, 5) == "hello") {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: if (str.starts_with("hello")) {}

  if ("hello" == str.substr(0, 5)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: if (str.starts_with("hello")) {}

  bool b = str.substr(0, 5) != "hello";
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: bool b = !str.starts_with("hello");

  // Variable length and string refs
  std::string prefix = "hello";
  size_t len = 5;
  if (str.substr(0, len) == prefix) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: if (str.starts_with(prefix)) {}

  // Various zero expressions
  const int zero = 0;
  int i = 0;
  if (str.substr(zero, 5) == "hello") {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: if (str.starts_with("hello")) {}

  if (str.substr(i-i, 5) == "hello") {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use starts_with() instead of substring comparison
  // CHECK-FIXES: if (str.starts_with("hello")) {}

  // Should not convert these
  if (str.substr(1, 5) == "hello") {}  // Non-zero start
  if (str.substr(0, 4) == "hello") {}  // Length mismatch
  if (str.substr(0, 6) == "hello") {}  // Length mismatch
}
