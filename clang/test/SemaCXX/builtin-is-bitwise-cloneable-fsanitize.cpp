// RUN: %clang_cc1 -triple x86_64-unknown-linux -DSANITIZER_ENABLED -fsanitize=address -fsanitize-address-field-padding=1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s

struct S {
  ~S() {}
  virtual void foo() {}

  int buffer[1];
  int other_field = 0;
};

static_assert(!__is_trivially_copyable(S));
#ifdef SANITIZER_ENABLED
static_assert(sizeof(S) > 16);
// Don't allow memcpy when the struct has poisoned padding bits.
static_assert(!__is_bitwise_cloneable(S));
#else
static_assert(sizeof(S) == 16);
static_assert(__is_bitwise_cloneable(S));
#endif
