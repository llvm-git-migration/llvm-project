// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
void foo() {
         //     expected-error@+1    2      {{use of undeclared identifier 'a'}}
    a = 2; a = 2; b = 2; b = 2; c = 2;
         //     expected-error@+1    2      {{asdf}}
    d = 2;
    e = 2; f = 2;                 //     expected-error    2      {{use of undeclared identifier 'e'}}
}

