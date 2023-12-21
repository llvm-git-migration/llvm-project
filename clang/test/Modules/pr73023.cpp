// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cpp -I%t -fsyntax-only -verify

//--- i.h
#ifndef FOO_H
#pragma once
struct S{};
#endif

//--- foo.cpp
// expected-no-diagnostics
#include "i.h"
#include "i.h"

int foo() {
    return sizeof(S);
}
