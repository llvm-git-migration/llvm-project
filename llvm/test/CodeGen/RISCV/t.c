void bar();
void foo1();
void foo2();
void bat();

void foo(short *x) {
  short y = *x;
  if (y > 15) {
  switch (y) {
  case 16:
    bar();
    break;
  case 17:
    foo1();
  case 18:
    foo2();
    break;
  }
  }

  bat();
}
