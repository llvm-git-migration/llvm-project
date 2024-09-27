// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=expected,both -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=ref,both

static_assert((bool)1.0k);
static_assert(!((bool)0.0k));
static_assert((bool)0.0k); // both-error {{static assertion failed}}
