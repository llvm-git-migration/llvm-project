// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
void foo() {
    a = 2;
    // expected-error@-1{{use of undeclared identifier 'a'}}
    b = 2;// expected-error{{use of undeclared identifier 'b'}}
    c = 2;
    // expected-error@7{{use of undeclared identifier 'c'}}
    d = 2; // expected-error-re{{use of {{.*}} identifier 'd'}}

    e = 2; // error to trigger mismatch
}

