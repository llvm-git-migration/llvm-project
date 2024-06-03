struct A {
  char a;
  char b;
  long int c;
  char d;
  int e;
  A() {};
};

struct B {
  double x;
  double y;
  B() {};
};

struct C {
  A a;
  char z;
  B b;
  C() {};
};

int main(int argc, char **argv) {
  long int acc = 0;

  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;

  C *c = new C();
  acc += c->a.a;
  acc += c->a.a;
  acc += c->b.x;
  acc += c->b.y;

  return 0;
}
