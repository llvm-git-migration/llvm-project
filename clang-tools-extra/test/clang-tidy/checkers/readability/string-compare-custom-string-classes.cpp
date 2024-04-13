// RUN: %check_clang_tidy %s readability-string-compare %t -- -config='{CheckOptions: {readability-string-compare.StringClassNames: "::std::basic_string;CustomStringTemplateBase;CustomStringNonTemplateBase"}}' -- -isystem %clang_tidy_headers
#include <string>

struct CustomStringNonTemplateBase {
  int compare(const CustomStringNonTemplateBase& Other) const {
    return 123;  // value is not important for check
  }
};

template <typename T>
struct CustomStringTemplateBase {
  int compare(const CustomStringTemplateBase& Other) const {
    return 123;
  }
};

struct DerivedFromStdString : std::string {};
struct CustomString1 : CustomStringNonTemplateBase {};
struct CustomString2 : CustomStringTemplateBase<char> {};

void CustomStringClasses() {
  std::string_view sv1("a");
  std::string_view sv2("b");
  if (sv1.compare(sv2)) {  // No warning - StringClassNames does not contain string_view.
  }

  DerivedFromStdString derived;
  if (derived.compare(derived)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings; use the string equality operator instead [readability-string-compare]

  CustomString1 custom1;
  if (custom1.compare(custom1)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings; use the string equality operator instead [readability-string-compare]

  CustomString2 custom2;
  if (custom2.compare(custom2)) {
  }
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: do not use 'compare' to test equality of strings; use the string equality operator instead [readability-string-compare]
}
