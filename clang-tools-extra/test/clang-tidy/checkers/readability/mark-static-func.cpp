// RUN: %check_clang_tidy %s readability-mark-static %t -- -- -I%S/Inputs/mark-static-var

#include "func.h"

void func() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func'

template<class T>
void func_template() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_template'

struct S {
  void method();
};
void S::method() {}

void func_header();
extern void func_extern();
static void func_static();
namespace {
void func_anonymous_ns();
} // namespace

int main(int argc, const char*argv[]) {}
