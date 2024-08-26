// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx11 -pedantic -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx14 -pedantic -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 -pedantic -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -pedantic -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -pedantic -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx26 -pedantic -std=c++26 %s

#define noreturn 1           // cxx11-warning {{keyword is hidden by macro definition}}
#undef noreturn

#define carries_dependency 1 // cxx11-warning {{keyword is hidden by macro definition}}
#undef carries_dependency

#define deprecated 1         // cxx14-warning {{keyword is hidden by macro definition}}
#undef deprecated

#define fallthrough 1        // cxx17-warning {{keyword is hidden by macro definition}}
#undef fallthrough

#define maybe_unused 1       // cxx17-warning {{keyword is hidden by macro definition}}
#undef maybe_unused

#define nodiscard 1          // cxx17-warning {{keyword is hidden by macro definition}}
#undef nodiscard

#define no_unique_address 1  // cxx20-warning {{keyword is hidden by macro definition}}
#undef no_unique_address

#define assume 1             // cxx23-warning {{keyword is hidden by macro definition}}
#undef assume

#define indeterminate 1      // cxx26-warning {{keyword is hidden by macro definition}}
#undef indeterminate
