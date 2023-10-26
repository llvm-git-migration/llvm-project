// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -Wshift-count-negative %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp -Wshift-count-negative %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp -Wall %s

enum shiftof {
    X = (1<<-29) // c-error {{expression is not an integer constant expression}}
                 // cpp-error@-1 {{expression is not an integral constant expression}}
                 // expected-note@-2 {{negative shift count -29}}
};
