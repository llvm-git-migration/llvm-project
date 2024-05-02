// RUN: %check_clang_tidy %s readability-mark-static %t -- -- -I%S/Inputs/mark-static-var

#include "var.h"

int global;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'global'

template<class T>
T global_template;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'global_template'

int gloabl_header;

extern int global_extern;

static int global_static;

namespace {
static int global_anonymous_ns;
namespace NS {
static int global_anonymous_ns;
}
}

static void f(int para) {
  int local;
  static int local_static;
}

struct S {
  int m1;
  static int m2;
};
int S::m2;
