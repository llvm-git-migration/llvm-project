// RUN: %clang_cc1 -fsyntax-only -verify %s

void func(int n) {
    int grp[n][n];
    int (*ptr)[n];

    for (int i = 0; i < n; i++)
        ptr = &grp[i]; // expected-error {{incompatible pointer types assigning to 'int (*)[n]' from 'int (*)[n]'; consider using a typedef to use the same variable-length array type for both operands}}
}
