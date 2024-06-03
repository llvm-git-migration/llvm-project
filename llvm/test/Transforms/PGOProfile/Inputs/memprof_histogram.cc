
struct A {
  long int a;
  long int b;
  long int c;
  long int d;
  long int e;
  long int f;
  long int g;
  long int h;
  A() {};
};

void foo() {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;
  acc += a->f;
  acc += a->g;
  acc += a->h;
  delete a;
}
void bar() {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->e;
  acc += a->e;
  acc += a->e;
  acc += a->e;
  acc += a->f;
  acc += a->f;
  acc += a->f;
  acc += a->g;
  acc += a->g;
  acc += a->h;

  delete a;
}

int main(int argc, char **argv) {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;
  acc += a->f;
  acc += a->g;
  acc += a->h;

  delete a;

  A *b = new A();
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->e;
  acc += b->e;
  acc += b->e;
  acc += b->e;
  acc += b->f;
  acc += b->f;
  acc += b->f;
  acc += b->g;
  acc += b->g;
  acc += b->h;

  delete b;

  A *c = new A();
  acc += c->a;

  for (int i = 0; i < 21; ++i) {

    foo();
  }

  for (int i = 0; i < 21; ++i) {

    bar();
  }

  return 0;
}
