// RUN: %clang_cc1 %s -verify=c -fsyntax-only -fflex-array-extensions

// The test checks that flexible array members do not emit warnings when
// -fflex-array-extensions when used in a union or alone in a structure.

struct already_hidden {
	int a;
	union {
		int b;
		struct {
			struct { } __empty;
			char array[];
		};
	};
};

struct still_zero_sized {
	struct { } __unused;
	int array[];
};

struct no_warn1 {
	int a;
	union {
		int b;
		char array[];
	};
};

struct no_warn2 {
	int array[];
};

union no_warn3 {
	short array[];
};

struct still_illegal {
	int array[]; // c-error {{flexible array member 'array' with type 'int[]' is not at the end of struct}}
	int a;       // c-note {{next field declaration is here}}
};

// expected-no-diagnostics
