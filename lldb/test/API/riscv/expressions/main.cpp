struct S {
  int a;
  int b;
};

struct U {
  int a;
  double d;
};

int g;

char *g_str = "global";

int func_with_double_arg(int a, double b) { return 1; }

int func_with_float_arg(float a, float b) { return 7; }
float func_with_float_ret_val() { return 8.0f; }

int func_with_ptr_arg(char *msg) { return 2; }
char *func_with_ptr_return() { return g_str; }
char *func_with_ptr(char *msg) { return msg; }

int func_with_struct_arg(S s) { return s.a + s.b; }

int func_with_double_struct_arg(U u) { return u.a * 3 + 31; }

double func_with_double_return() { return 42.0; }

S func_with_struct_return() {
  S s = {123, 4};
  return s;
}

U func_with_double_struct_return() {
  U u = {123, 42.0};
  return u;
}

int foo() { return 3; }

int foo(int a) { return a; }

int foo(int a, int b) { return a + b; }

int main() {
  S s{11, 99};
  U u{123, 7.7};

  double d = func_with_double_arg(1, 1.0) + func_with_struct_arg(S{1, 2}) +
             func_with_ptr_arg("msg") + func_with_double_return() +
             func_with_double_struct_arg(U{1, 1.0}) + foo() + foo(1) +
             foo(1, 2) + func_with_float_arg(5.42, 7.86) +
             func_with_float_ret_val();

  char *msg = func_with_ptr("msg");
  char *ptr = func_with_ptr_return();

  S s_s = func_with_struct_return();
  U s_u = func_with_double_struct_return(); // break here

  return 0;
}
