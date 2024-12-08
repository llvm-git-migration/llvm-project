// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify

// expected-warning@+2 {{incompatible redeclaration of library function 'fabs'}}
// expected-note@+1 {{'fabs' is a builtin with type 'double (double)'}}
extern __inline  double
fabs (char  __x)
{
  register double __value;
  __asm __volatile__
    ("fabs"
     : "=t" (__value) : "0" (__x));  // expected-error {{unsupported inline asm: input with type 'char' matching output with type 'double'}}
  return __value;
}
int
foo ()
{
  int i, j, k;
  double x = 0, y = ((i == j) ? 1 : 0);
  for (i = 0; i < 10; i++)
    ;
  fabs (x - y);
  return 0;
}