// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
//
struct DynamicClass { virtual int Foo(); };
static_assert(!__is_trivially_copyable(DynamicClass));
static_assert(__is_bitwise_cloneable(DynamicClass));

struct InComplete;
static_assert(!__is_bitwise_cloneable(InComplete));
