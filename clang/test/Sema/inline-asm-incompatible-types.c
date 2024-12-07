extern __inline  double
fabs (char  __x)
{
  register double __value;
  __asm __volatile__
    ("fabs"
     : "=t" (__value) : "0" (__x));
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