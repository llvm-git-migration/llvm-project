// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -x c++
// RUN: %clang_cc1 %s -verify -fsyntax-only -fms-compatibility -x c++
// RUN: %clang_cc1 %s -verify=gnu -fsyntax-only -Wgnu-flexible-array-union-member -Wgnu-empty-struct
// RUN: %clang_cc1 %s -verify=microsoft -fsyntax-only -fms-compatibility -Wmicrosoft

// The test checks that an attempt to initialize union with flexible array
// member with an initializer list doesn't crash clang.


union { char x[]; } r = {0}; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                                microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
                              */

struct already_hidden {
  int a;
  union {
    int b;
    struct {
      struct { } __empty;  // gnu-warning {{empty struct is a GNU extension}}
      char x[];
    };
  };
};

struct still_zero_sized {
  struct { } __unused;  // gnu-warning {{empty struct is a GNU extension}}
  int x[];
};

struct warn1 {
  int a;
  union {
    int b;
    char x[]; /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
  };
};

struct warn2 {
  int x[];  /* gnu-warning {{flexible array member 'x' in otherwise empty struct is a GNU extension}}
               microsoft-warning {{flexible array member 'x' in otherwise empty struct is a Microsoft extension}}
             */
};

union warn3 {
  short x[];  /* gnu-warning {{flexible array member 'x' in a union is a GNU extension}}
                 microsoft-warning {{flexible array member 'x' in a union is a Microsoft extension}}
               */
};

struct quiet1 {
  int a;
  short x[];
};

// expected-no-diagnostics
