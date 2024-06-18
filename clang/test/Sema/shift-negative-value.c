// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -Wshift-negative-value %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp -Wshift-negative-value %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp -Wall %s

enum shiftof {
    X = (-1<<29) // c-error {{expression is not an integer constant expression}}
                 // cpp-error@-1 {{expression is not an integral constant expression}}
                 // expected-note@-2 {{left shift of negative value -1}}
};
