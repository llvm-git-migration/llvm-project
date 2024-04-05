// RUN: %clang_cc1 -fsyntax-only -verify=expected,c_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -verify=expected,cpp_diagnostics -std=c++23 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#ifndef __cplusplus
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
typedef __WCHAR_TYPE__ wchar_t;
#endif

typedef __SIZE_TYPE__ size_t;
typedef __builtin_va_list va_list;

__attribute__((__format__(__printf__, 1, 2)))
int printf(const char *, ...); // #printf

__attribute__((__format__(__scanf__, 1, 2)))
int scanf(const char *, ...); // #scanf

__attribute__((__format__(__printf__, 1, 0)))
int vprintf(const char *, va_list); // #vprintf

__attribute__((__format__(__scanf__, 1, 0)))
int vscanf(const char *, va_list); // #vscanf

__attribute__((__format__(__printf__, 2, 0)))
int vsprintf(char *, const char *, va_list); // #vsprintf

__attribute__((__format__(__printf__, 3, 0)))
int vsnprintf(char *ch, size_t, const char *, va_list); // #vsnprintf

#ifndef __cplusplus
int vwscanf(const wchar_t *, va_list); // #vwscanf
#endif

__attribute__((__format__(__scanf__, 1, 4)))
void f1(char *out, const size_t len, const char *format, ... /* args */) // #f1
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-no-warning@#f1
}

__attribute__((__format__(__printf__, 1, 4)))
void f2(char *out, const size_t len, const char *format, ... /* args */) // #f2
{
    va_list args;
    vsnprintf(out, len, format, args); // expected-warning@#f2 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f2'}}
                                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 3, 4)))"
}

void f3(char *out, va_list args) // #f3
{
    vprintf(out, args); // expected-warning@#f3 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f3'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f4(char* out, ... /* args */) // #f4
{
    va_list args;
    vprintf("test", args); // expected-no-warning@#f4

    const char *ch;
    vprintf(ch, args); // expected-no-warning@#f4
}

void f5(va_list args) // #f5
{
    char *ch;
    vscanf(ch, args); // expected-no-warning@#f5
}

void f6(char *out, va_list args) // #f6
{
    char *ch;
    vprintf(ch, args); // expected-no-warning@#f6
    vprintf("test", args); // expected-no-warning@#f6
    vprintf(out, args); // expected-warning@#f6 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f6'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f7(const char *out, ... /* args */) // #f7
{
    va_list args;

    vscanf(out, args); // expected-warning@#f7 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f7'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:6-[[@LINE-5]]:6}:"__attribute__((format(scanf, 1, 2)))"
}

void f8(const char *out, ... /* args */) // #f8
{
    va_list args;

    vscanf(out, args); // expected-no-warning@#f8
    vprintf(out, args); // expected-no-warning@#f8
}

void f9(const char out[], ... /* args */) // #f9
{
    va_list args;
    char *ch;
    vprintf(ch, args); // expected-no-warning
    vsprintf(ch, out, args); // expected-warning@#f9 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f9'}}
                             // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 2)))"
}

#ifndef __cplusplus
void f10(const wchar_t *out, ... /* args */) // #f10
{
    va_list args;
    vwscanf(out, args); // expected-no-warning@#f10
}
#endif

void f11(const char *out) // #f11
{
    va_list args;
    vscanf(out, args); // expected-warning@#f11 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f11'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f12(char* out) // #f12
{
    va_list args;
    const char* ch;
    vsprintf(out, ch, args); // expected-no-warning@#f12
    vprintf(out, args); // expected-warning@#f12 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f12'}}
                        // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f13(const char *out, ... /* args */) // #f13
{
    int a;
    printf(out, a); // expected-warning@#f13 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f13'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f14(const char *out, ... /* args */) // #f14
{
    printf(out, 1); // expected-warning@#f14 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f14'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(printf, 1, 0)))"
}

__attribute__((format(printf, 1, 2)))
void f15(const char *out, ... /* args */) // #f15
{
    int a;
    printf(out, a); // expected-no-warning@#f15
}

__attribute__((format(printf, 1, 2)))
void f16(const char *out, ... /* args */) // #f16
{
    printf(out, 1); // expected-no-warning@#f16
}

__attribute__((format(printf, 1, 2)))
void f17(const char *out, ... /* args */) // #f17
{
    int a;
    printf(out, a); // expected-no-warning@#f17
    printf(out, 1); // expected-no-warning@#f17
}

void f18(char *out, ... /* args */) // #f18
{
    va_list args;
    scanf(out, args); // expected-no-warning@#f18
    {
        printf(out, args); // expected-no-warning@#f18
    }
}

void f19(char *out, va_list args) // #f19
{
    {
        scanf(out, args); // expected-no-warning@#f19
        printf(out, args); // expected-no-warning@#f19
    }
}

// expected-warning@#f20 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f20'}}
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:6-[[@LINE+1]]:6}:"__attribute__((format(scanf, 1, 2)))"
void f20(char *out, ... /* args */) // #f20
{
    va_list args;
    scanf(out, args);
    {
        scanf(out, args);
    }
}

void f21(char* ch, const char *out, ... /* args */) // #f21
{
    va_list args;
    printf(ch, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 3)))"
    int a;
    printf(out, a); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:6-[[@LINE-7]]:6}:"__attribute__((format(printf, 2, 0)))"
    printf(out, 1); // no warning because first command above emitted same warning and fix-it
    printf(out, args); // expected-warning@#f21 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f21'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:6-[[@LINE-10]]:6}:"__attribute__((format(printf, 2, 3)))"
}

typedef va_list tdVaList;
typedef int tdInt;

void f22(const char *out, ... /* args */) // #f22
{
    tdVaList args;
    printf(out, args); // expected-warning@#f22 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f22'}}
                       // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 2)))"
}

void f23(const char *out, ... /* args */) // #f23
{
    tdInt a;
    scanf(out, a); // expected-warning@#f23 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f23'}}
                   // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f24(const char *out, tdVaList args) // #f24
{
    scanf(out, args); // expected-warning@#f24 {{diagnostic behavior may be improved by adding the 'scanf' format attribute to the declaration of 'f24'}}
                      // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:6-[[@LINE-3]]:6}:"__attribute__((format(scanf, 1, 0)))"
}

void f25(const char *out, tdVaList args) // #f25
{
    tdInt a;
    printf(out, a); // expected-warning@#f25 {{diagnostic behavior may be improved by adding the 'printf' format attribute to the declaration of 'f25'}}
                    // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:6-[[@LINE-4]]:6}:"__attribute__((format(printf, 1, 0)))"
}

void f26(char *out, ... /* args */) // #f26
{
    va_list args;
    char *ch;
    vscanf(ch, args); // expected-no-warning@#f26
    vprintf(out, args); // expected-no-warning@#f26
}

void f27(char *out, ... /* args */) // #f27
{
    va_list args;
    vscanf("%s", args); // expected-no-warning@#f27
    vprintf(out, args); // expected-no-warning@#f27
}
