// RUN: %clang_cc1 -fsyntax-only -verify -Wshift-count-overflow %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

enum shiftof {
    X = (1<<32) // expected-error {{expression is not an integer constant expression}}
                // expected-note@-1 {{shift count 32 >= width of type 'int'}}
};
