// RUN: %clang_cc1 -verify=check %s
// RUN: diff %s %s.expected
void foo() {
    a = 2; // check-error{{asdf}}
           // expected-error@-1{ignored}}
}

