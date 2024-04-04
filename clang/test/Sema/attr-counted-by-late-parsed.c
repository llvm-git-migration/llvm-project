// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_unknown;

struct at_pointer {
  int count;
  // FIXME: Diagnostic missing
  struct size_unknown * buf __counted_by(count); // expected-error{{'counted_by' cannot be applied to a sized type}}
};

struct size_known { int dummy; };

struct at_nested_pointer {
  // TODO: Support attribute late parsing in type attribute position.
  struct size_known *__counted_by(count) *buf; // expected-error{{use of undeclared identifier 'count'}}
  int count;
};

struct at_decl {
  struct size_known *buf __counted_by(count);
  int count;
};

struct at_pointer_anon_buf {
  struct {
    // TODO: Support referring to parent scope
    // TODO: Support attribute late parsing in type attribute position.
    struct size_known *__counted_by(count) buf; // expected-error{{use of undeclared identifier 'count'}}
  };
  int count;
};

struct at_decl_anon_buf {
  struct {
    // TODO: Support referring to nested scope
    struct size_known *buf __counted_by(count); // expected-error{{use of undeclared identifier 'count'}}
  };
  int count;
};

struct at_pointer_anon_count {
  // TODO: Support attribute late parsing in type attribute position.
  struct size_known *__counted_by(count) buf; // expected-error{{use of undeclared identifier 'count'}}
  struct {
    int count;
  };
};

struct at_decl_anon_count {
  struct size_known *buf __counted_by(count);
  struct {
    int count;
  };
};
