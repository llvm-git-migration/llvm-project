// RUN: not %clang_cc1 -fsyntax-only %s -fno-crash-diagnostics

a() {struct b c (sizeof(b * [({ {tree->d* next)} 0
