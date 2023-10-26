// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

enum shiftof {
    X = (1<<32) // expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                // expected-note@-1 {{shift count 32 >= width of type 'int'}}
};
