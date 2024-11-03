// RUN: %check_clang_tidy %s misc-redundant-expression %t -- -- -fno-delayed-template-parsing

typedef __INT64_TYPE__ I64;

struct Point {
  int x;
  int y;
  int a[5];
} P;

extern Point P1;
extern Point P2;

extern int foo(int x);
extern int bar(int x);
extern int bat(int x, int y);

int TestSimpleEquivalent(int X, int Y) {
  if (X - X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent [misc-redundant-expression]
  if (X / X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X % X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X & X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X | X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X ^ X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X < X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X <= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X > X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X >= X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent
  if (X || X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X != (((X)))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of operator are equivalent

  if (X + 1 == X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 != X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 <= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X + 1 >= X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent

  if ((X != 1 || Y != 1) && (X != 1 || Y != 1)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: both sides of operator are equivalent
  if (P.a[X - P.x] != P.a[X - P.x]) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: both sides of operator are equivalent

  if ((int)X < (int)X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent
  if (int(X) < int(X)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent

  if ( + "dummy" == + "dummy") return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent
  if (L"abc" == L"abc") return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent

  if (foo(0) - 2 < foo(0) - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent
  if (foo(bar(0)) < (foo(bar((0))))) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both sides of operator are equivalent

  if (P1.x < P2.x && P1.x < P2.x) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both sides of operator are equivalent
  if (P2.a[P1.x + 2] < P2.x && P2.a[(P1.x) + (2)] < (P2.x)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: both sides of operator are equivalent

  if (X && Y && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: operator has equivalent nested operands
  if (X || (Y || X)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: operator has equivalent nested operands
  if ((X ^ Y) ^ (Y ^ X)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: operator has equivalent nested operands

  return 0;
}

template <int DX>
int TestSimpleEquivalentDependent() {
  if (DX > 0 && DX > 0) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent

  return 0;
}

int Valid(int X, int Y) {
  if (X != Y) return 1;
  if (X == Y + 0) return 1;
  if (P.x == P.y) return 1;
  if (P.a[P.x] < P.a[P.y]) return 1;
  if (P.a[0] < P.a[1]) return 1;

  if (P.a[0] < P.a[0ULL]) return 1;
  if (0 < 0ULL) return 1;
  if ((int)0 < (int)0ULL) return 1;

  if (++X != ++X) return 1;
  if (P.a[X]++ != P.a[X]++) return 1;
  if (P.a[X++] != P.a[X++]) return 1;
  if (X && X++ && X) return 1;

  if ("abc" == "ABC") return 1;
  if (foo(bar(0)) < (foo(bat(0, 1)))) return 1;
  return 0;
}

#define COND_OP_MACRO 9
#define COND_OP_OTHER_MACRO 9
#define COND_OP_THIRD_MACRO COND_OP_MACRO
int TestConditional(int x, int y) {
  int k = 0;
  k += (y < 0) ? x : x;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? x + 1 : x + 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? COND_OP_MACRO : COND_OP_MACRO;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? COND_OP_MACRO + COND_OP_OTHER_MACRO : COND_OP_MACRO + COND_OP_OTHER_MACRO;
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: 'true' and 'false' expressions are equivalent

  // Do not match for conditional operators with a macro and a const.
  k += (y < 0) ? COND_OP_MACRO : 9;
  // Do not match for conditional operators with expressions from different macros.
  k += (y < 0) ? COND_OP_MACRO : COND_OP_OTHER_MACRO;
  // Do not match for conditional operators when a macro is defined to another macro
  k += (y < 0) ? COND_OP_MACRO : COND_OP_THIRD_MACRO;
#undef COND_OP_THIRD_MACRO
#define   COND_OP_THIRD_MACRO 8
  k += (y < 0) ? COND_OP_MACRO : COND_OP_THIRD_MACRO;
#undef COND_OP_THIRD_MACRO

  k += (y < 0) ? sizeof(I64) : sizeof(I64);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 'true' and 'false' expressions are equivalent
  k += (y < 0) ? sizeof(TestConditional(k,y)) : sizeof(TestConditional(k,y));
  // CHECK-MESSAGES: :[[@LINE-1]]:47: warning: 'true' and 'false' expressions are equivalent
  // No warning if the expression arguments are different.
  k += (y < 0) ? sizeof(TestConditional(k,y)) : sizeof(Valid(k,y));

  return k;
}
#undef COND_OP_MACRO
#undef COND_OP_OTHER_MACRO

// Overloaded operators that compare two instances of a struct.
struct MyStruct {
  int x;
  bool operator==(const MyStruct& rhs) const {return this->x == rhs.x; } // not modifing
  bool operator>=(const MyStruct& rhs) const { return this->x >= rhs.x; } // not modifing
  bool operator<=(MyStruct& rhs) const { return this->x <= rhs.x; }
  bool operator&&(const MyStruct& rhs){ this->x++; return this->x && rhs.x; }
} Q;

bool operator!=(const MyStruct& lhs, const MyStruct& rhs) { return lhs.x == rhs.x; } // not modifing
bool operator<(const MyStruct& lhs, const MyStruct& rhs) { return lhs.x < rhs.x; } // not modifing
bool operator>(const MyStruct& lhs, MyStruct& rhs) { rhs.x--; return lhs.x > rhs.x; }
bool operator||(MyStruct& lhs, const MyStruct& rhs) { lhs.x++; return lhs.x || rhs.x; }

struct MyStruct1 {
  bool x;
  MyStruct1(bool x) : x(x) {};
  operator bool() { return x; }
};

MyStruct1 operator&&(const MyStruct1& lhs, const MyStruct1& rhs) { return lhs.x && rhs.x; }
MyStruct1 operator||(MyStruct1& lhs, MyStruct1& rhs) { return lhs.x && rhs.x; }

bool TestOverloadedOperator(MyStruct& S) {
  if (S == Q) return false;

  if (S <= S) return false;
  if (S && S) return false;
  if (S > S) return false;
  if (S || S) return false;

  if (S == S) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of overloaded operator are equivalent
  if (S < S) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of overloaded operator are equivalent
  if (S != S) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of overloaded operator are equivalent
  if (S >= S) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: both sides of overloaded operator are equivalent

  MyStruct1 U(false);
  MyStruct1 V(true);

  // valid because the operator is not const
  if ((U || V) || U) return true;

  if (U && V && U && V) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: overloaded operator has equivalent nested operands

  return true;
}

#define LT(x, y) (void)((x) < (y))
#define COND(x, y, z) ((x)?(y):(z))
#define EQUALS(x, y) (x) == (y)

int TestMacro(int X, int Y) {
  LT(0, 0);
  LT(1, 0);
  LT(X, X);
  LT(X+1, X + 1);
  COND(X < Y, X, X);
  EQUALS(Q, Q);
  return 0;
}

int TestFalsePositive(int* A, int X, float F) {
  // Produced by bison.
  X = A[(2) - (2)];
  X = A['a' - 'a'];

  // Testing NaN.
  if (F != F && F == F) return 1;
  return 0;
}

int TestBannedMacros() {
#define EAGAIN 3
#define NOT_EAGAIN 3
  if (EAGAIN == 0 | EAGAIN == 0) return 0;
  if (NOT_EAGAIN == 0 | NOT_EAGAIN == 0) return 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: both sides of operator are equivalent
  return 0;
}

struct MyClass {
static const int Value = 42;
};
template <typename T, typename U>
void TemplateCheck() {
  static_assert(T::Value == U::Value, "should be identical");
  static_assert(T::Value == T::Value, "should be identical");
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: both sides of operator are equivalent
}
void TestTemplate() { TemplateCheck<MyClass, MyClass>(); }

int TestArithmetic(int X, int Y) {
  if (X + 1 == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X - 1 == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X - 1 != X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1LL == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X + 1ULL == X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false

  if (X == X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false
  if (X != X + 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true
  if (X == X - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false
  if (X != X - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true

  if (X != X - 1U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true
  if (X != X - 1LL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always true

  if ((X+X) != (X+X) - 1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X + 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X + 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X - 1 == X - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X - 1 != X - 2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - -1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X + 1 != X - -1) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 == X - -2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X + 1 != X - -2) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - (~0)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true
  if (X + 1 == X - (~0U)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if (X + 1 == X - (~0ULL)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  // Should not match.
  if (X + 0.5 == X) return 1;
  if (X + 1 == Y) return 1;
  if (X + 1 == Y + 1) return 1;
  if (X + 1 == Y + 2) return 1;

  return 0;
}

int TestBitwise(int X, int Y) {

  if ((X & 0xFF) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((X & 0xFF) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true
  if ((X | 0xFF) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((X | 0xFF) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((X | 0xFFULL) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: logical expression is always true
  if ((X | 0xFF) != 0xF00ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((0xFF & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((0xFF & X) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true
  if ((0xFF & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if ((0xFF & X) != 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always true

  if ((0xFFLL & X) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: logical expression is always false
  if ((0xFF & X) == 0xF00ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  return 0;
}

// Overloaded operators that compare an instance of a struct and an integer
// constant.
struct S {
  S() { x = 1; }
  int x;
  // Overloaded comparison operators without any possible side effect.
  bool operator==(const int &i) const { return x == i; } // not modifying
  bool operator!=(int i) const { return x != i; } // not modifying
  bool operator>(const int &i) const { return x > i; } // not modifying
  bool operator<(int i) const { return x < i; } // not modifying
};

bool operator<=(const S &s, int i) { return s.x <= i; } // not modifying
bool operator>=(const S &s, const int &i) { return s.x >= i; } // not modifying

bool operator==(int i, const S &s) { return s == i; } // not modifying
bool operator<(const int &i, const S &s) { return s > i; } // not modifying
bool operator<=(const int &i, const S &s) { return s >= i; } // not modifying
bool operator>(const int &i, const S &s) { return s < i; } // not modifying

struct S2 {
  S2() { x = 1; }
  int x;
  // Overloaded comparison operators that are able to modify their params.
  bool operator==(const int &i) {
    this->x++;
    return x == i;
  }
  bool operator!=(int i) { return x != i; }
  bool operator>(const int &i) { return x > i; }
  bool operator<(int i) {
    this->x--;
    return x < i;
  }
};

bool operator>=(S2 &s, const int &i) { return s.x >= i; }
bool operator<=(S2 &s, int i) {
  s.x++;
  return s.x <= i;
}

int TestLogical(int X, int Y){
#define CONFIG 0
  if (CONFIG && X) return 1;
#undef CONFIG
#define CONFIG 1
  if (CONFIG || X) return 1;
#undef CONFIG

  if (X == 10 && X != 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && (X != 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (!(X != 10) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  if (X == 10ULL && X != 10ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false
  if (!(X != 10U) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false
  if (!(X != 10LL) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: logical expression is always false
  if (!(X != 10ULL) && !(X == 10)) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: logical expression is always false

  if (X == 0 && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X != 0 && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: logical expression is always false

  if (X && !!X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: equivalent expression on both sides of logical operator
  if (X != 0 && X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X != 0 && !!X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X == 0 && !X) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator

  // Should not match.
  if (X == 10 && Y == 10) return 1;
  if (X != 10 && X != 12) return 1;
  if (X == 10 || X == 12) return 1;
  if (!X && !Y) return 1;
  if (!X && Y) return 1;
  if (!X && Y == 0) return 1;
  if (X == 10 && Y != 10) return 1;

  // Test for overloaded operators with constant params.
  S s1;
  if (s1 == 1 && s1 == 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: equivalent expression on both sides of logical operator
  if (s1 == 1 || s1 != 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (s1 > 1 && s1 < 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (s1 >= 1 || s1 <= 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (s1 >= 2 && s1 <= 0) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false

  // Same test as above but with swapped LHS/RHS on one side of the logical operator.
  if (1 == s1 && s1 == 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: equivalent expression on both sides of logical operator
  if (1 == s1 || s1 != 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (1 < s1 && s1 < 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (1 <= s1 || s1 <= 1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (2 < s1 && 0 > s1) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false

  // Test for absence of false positives (issue #54011).
  if (s1 == 1 || s1 == 2) return true;
  if (s1 > 1 && s1 < 3) return true;
  if (s1 >= 2 || s1 <= 0) return true;

  // Test for overloaded operators that may modify their params.
  S2 s2;
  if (s2 == 1 || s2 != 1) return true;
  if (s2 == 1 || s2 == 1) return true;
  if (s2 > 1 && s2 < 1) return true;
  if (s2 >= 1 || s2 <= 1) return true;
}

int TestRelational(int X, int Y) {
  if (X == 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X == 10 && X < 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X < 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X <= 10 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always false
  if (X < 10 && X >= 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false
  if (X < 10 && X == 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false

  if (X > 5 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always false
  if (X > -5 && X <= -5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always false

  if (X < 10 || X >= 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true
  if (X <= 10 || X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (X <= 10 || X >= 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: logical expression is always true
  if (X != 7 || X != 14) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true
  if (X == 7 || X != 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 7 || X == 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: logical expression is always true

  if (X < 7 && X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 7 && X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X < 7 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  if (X < 7 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 7 && X <= 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: equivalent expression on both sides of logical operator
  if (X < 7 && X <= 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant
  if (X < 7 && X <= 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  if (X <= 7 && X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator

  if (X >= 7 && X > 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: equivalent expression on both sides of logical operator
  if (X >= 7 && X > 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X >= 7 && X > 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X <= 7 && X <= 5) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X <= 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X <= 7 && X <= 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent
  if (X <= 7 && X <= 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: expression is redundant

  if (X == 11 && X > 10) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: expression is redundant
  if (X == 11 && X < 12) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: expression is redundant
  if (X > 10 && X == 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X < 12 && X == 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X != 11 && X == 42) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X > 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X < 11) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != 11 && X > 14) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X < 7 || X < 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant
  if (X < 7 || X < 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X < 7 || X < 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  if (X > 7 || X > 6) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X > 7 || X > 7) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent
  if (X > 7 || X > 8) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: expression is redundant

  // Should not match.
  if (X < 10 || X > 12) return 1;
  if (X > 10 && X < 12) return 1;
  if (X < 10 || X >= 12) return 1;
  if (X > 10 && X <= 12) return 1;
  if (X <= 10 || X > 12) return 1;
  if (X >= 10 && X < 12) return 1;
  if (X <= 10 || X >= 12) return 1;
  if (X >= 10 && X <= 12) return 1;
  if (X >= 10 && X <= 11) return 1;
  if (X >= 10 && X < 11) return 1;
  if (X > 10 && X <= 11) return 1;
  if (X > 10 && X != 11) return 1;
  if (X >= 10 && X <= 10) return 1;
  if (X <= 10 && X >= 10) return 1;
  if (X < 0 || X > 0) return 1;
}

int TestRelationalMacros(int X){
#define SOME_MACRO 3
#define SOME_MACRO_SAME_VALUE 3
#define SOME_OTHER_MACRO 9
  // Do not match for redundant relational macro expressions that can be
  // considered intentional, and for some particular values, non redundant.

  // Test cases for expressions with the same macro on both sides.
  if (X < SOME_MACRO && X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always false
  if (X < SOME_MACRO && X == SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always false
  if (X < SOME_MACRO || X >= SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always true
  if (X <= SOME_MACRO || X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: logical expression is always true
  if (X != SOME_MACRO && X > SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant
  if (X != SOME_MACRO && X < SOME_MACRO) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: expression is redundant

  // Test cases for two different macros.
  if (X < SOME_MACRO && X > SOME_OTHER_MACRO) return 1;
  if (X != SOME_MACRO && X >= SOME_OTHER_MACRO) return 1;
  if (X != SOME_MACRO && X != SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO || X == SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO || X <= SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO || X > SOME_MACRO_SAME_VALUE) return 1;
  if (X < SOME_MACRO && X <= SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X > SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X != SOME_OTHER_MACRO) return 1;
  if (X == SOME_MACRO && X != SOME_MACRO_SAME_VALUE) return 1;
  if (X == SOME_MACRO_SAME_VALUE && X == SOME_MACRO ) return 1;

  // Test cases for a macro and a const.
  if (X < SOME_MACRO && X > 9) return 1;
  if (X != SOME_MACRO && X >= 9) return 1;
  if (X != SOME_MACRO && X != 9) return 1;
  if (X == SOME_MACRO || X == 3) return 1;
  if (X == SOME_MACRO || X <= 3) return 1;
  if (X < SOME_MACRO && X <= 9) return 1;
  if (X == SOME_MACRO && X != 9) return 1;
  if (X == SOME_MACRO && X == 9) return 1;

#undef SOME_OTHER_MACRO
#undef SOME_MACRO_SAME_VALUE
#undef SOME_MACRO
  return 0;
}

int TestValidExpression(int X) {
  if (X - 1 == 1 - X) return 1;
  if (2 * X == X) return 1;
  if ((X << 1) == X) return 1;

  return 0;
}

enum Color { Red, Yellow, Green };
int TestRelationalWithEnum(enum Color C) {
  if (C == Red && C == Yellow) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false
  if (C == Red && C != Red) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always false
  if (C != Red || C != Yellow) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: logical expression is always true

  // Should not match.
  if (C == Red || C == Yellow) return 1;
  if (C != Red && C != Yellow) return 1;

  return 0;
}

template<class T>
int TestRelationalTemplated(int X) {
  // This test causes a corner case with |isIntegerConstantExpr| where the type
  // is dependent. There is an assert failing when evaluating
  // sizeof(<incomplet-type>).
  if (sizeof(T) == 4 || sizeof(T) == 8) return 1;

  if (X + 0 == -X) return 1;
  if (X + 0 < X) return 1;

  return 0;
}

int TestWithSignedUnsigned(int X) {
  if (X + 1 == X + 1ULL) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: logical expression is always true

  if ((X & 0xFFU) == 0xF00) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false

  if ((X & 0xFF) == 0xF00U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: logical expression is always false

  if ((X & 0xFFU) == 0xF00U) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: logical expression is always false

  return 0;
}

int TestWithLong(int X, I64 Y) {
  if (X + 0ULL == -X) return 1;
  if (Y + 0 == -Y) return 1;
  if (Y <= 10 && X >= 10LL) return 1;
  if (Y <= 10 && X >= 10ULL) return 1;
  if (X <= 10 || X > 12LL) return 1;
  if (X <= 10 || X > 12ULL) return 1;
  if (Y <= 10 || Y > 12) return 1;

  return 0;
}

int TestWithMinMaxInt(int X) {
  if (X <= X + 0xFFFFFFFFU) return 1;
  if (X <= X + 0x7FFFFFFF) return 1;
  if (X <= X + 0x80000000) return 1;

  if (X <= 0xFFFFFFFFU && X > 0) return 1;
  if (X <= 0xFFFFFFFFU && X > 0U) return 1;

  if (X + 0x80000000 == X - 0x80000000) return 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: logical expression is always true

  if (X > 0x7FFFFFFF || X < ((-0x7FFFFFFF)-1)) return 1;
  if (X <= 0x7FFFFFFF && X >= ((-0x7FFFFFFF)-1)) return 1;

  return 0;
}

#define FLAG1 1
#define FLAG2 2
#define FLAG3 4
#define FLAGS (FLAG1 | FLAG2 | FLAG3)
#define NOTFLAGS !(FLAG1 | FLAG2 | FLAG3)
int TestOperatorConfusion(int X, int Y, long Z)
{
  // Ineffective & expressions.
  Y = (Y << 8) & 0xff;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: ineffective bitwise and operation
  Y = (Y << 12) & 0xfff;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: ineffective bitwise and
  Y = (Y << 12) & 0xff;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: ineffective bitwise and
  Y = (Y << 8) & 0x77;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: ineffective bitwise and
  Y = (Y << 5) & 0x11;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: ineffective bitwise and

  // Tests for unmatched types
  Z = (Z << 8) & 0xff;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: ineffective bitwise and operation
  Y = (Y << 12) & 0xfffL;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: ineffective bitwise and
  Z = (Y << 12) & 0xffLL;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: ineffective bitwise and
  Y = (Z << 8L) & 0x77L;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: ineffective bitwise and

  Y = (Y << 8) & 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: ineffective bitwise and

  Y = (Y << 8) & -1;

  // Effective expressions. Do not check.
  Y = (Y << 4) & 0x15;
  Y = (Y << 3) & 0x250;
  Y = (Y << 9) & 0xF33;

  int K = !(1 | 2 | 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: ineffective logical negation operator used; did you mean '~'?
  // CHECK-FIXES: {{^}}  int K = ~(1 | 2 | 4);{{$}}
  K = !(FLAG1 & FLAG2 & FLAG3);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: ineffective logical negation operator
  // CHECK-FIXES: {{^}}  K = ~(FLAG1 & FLAG2 & FLAG3);{{$}}
  K = !(3 | 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: ineffective logical negation operator
  // CHECK-FIXES: {{^}}  K = ~(3 | 4);{{$}}
  int NotFlags = !FLAGS;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: ineffective logical negation operator
  // CHECK-FIXES: {{^}}  int NotFlags = ~FLAGS;{{$}}
  NotFlags = NOTFLAGS;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: ineffective logical negation operator
  return !(1 | 2 | 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: ineffective logical negation operator
  // CHECK-FIXES: {{^}}  return ~(1 | 2 | 4);{{$}}
}

template <int Shift, int Mask>
int TestOperatorConfusionDependent(int Y) {
  int r1 = (Y << Shift) & 0xff;
  int r2 = (Y << 8) & Mask;
}
#undef FLAG1
#undef FLAG2
#undef FLAG3

namespace no_crash {
struct Foo {};
bool operator<(const Foo&, const Foo&);
template <class T>
struct Bar {
  static const Foo &GetFoo();
  static bool Test(const T & maybe_foo, const Foo& foo) {
    return foo < GetFoo() && foo < maybe_foo;
  }
};

template <class... Values>
struct Bar2 {
  static_assert((... && (sizeof(Values) > 0)) == (... && (sizeof(Values) > 0)));
  // FIXME: It's not clear that we should be diagnosing this. The `&&` operator
  // here is unresolved and could resolve to an overloaded operator that might
  // have side-effects on its operands. For other constructs with the same
  // property (eg, the `S2` cases above) we suppress this diagnostic. This
  // started failing when Clang started properly modeling the fold-expression as
  // containing an unresolved operator name.
  // FIXME-MESSAGES: :[[@LINE-1]]:47: warning: both sides of operator are equivalent [misc-redundant-expression]
};

} // namespace no_crash

int TestAssignSideEffect(int i) {
  int k = i;

  if ((k = k + 1) != 1 || (k = k + 1) != 2)
    return 0;

  if ((k = foo(0)) != 1 || (k = foo(0)) != 2)
    return 1;

  return 2;
}

namespace PR63096 {

struct alignas(sizeof(int)) X {
  int x;
};

static_assert(alignof(X) == sizeof(X));
static_assert(sizeof(X) == sizeof(X));
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: both sides of operator are equivalent

}

namespace PR35857 {
  void test() {
    int x = 0;
    int y = 0;
    decltype(x + y - (x + y)) z = 10;
  }
}

// Ported from identical-expression

int func(void)
{
  return 0;
}

int func2(void)
{
  return 0;
}

int funcParam(int a)
{
  return 0;
}

/* '!=' operator*/

/* '!=' with float */
int checkNotEqualFloatLiteralCompare1(void) {
  return (5.14F != 5.14F); // no warning
}

int checkNotEqualFloatLiteralCompare2(void) {
  return (6.14F != 7.14F); // no warning
}

int checkNotEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f != g); // no warning
}

int checkNotEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f != f); // no warning
}

int checkNotEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f != 7.1F); // no warning
}

int checkNotEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F != f); // no warning
}

int checkNotEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t != f); // no warning
}

int checkNotEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f != t); // no warning
}



int checkNotEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f != (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F != f + 3.14F);  // no warning
  return (0);
}
int checkNotEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F != g + 3.14F); // no warning
}
int checkNotEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (int)f + 3.14F);  // no warning
  return (0);
}
int checkNotEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (char)f + 3.14F);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) != ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*(f + t != f + t)));  // no warning
  return (0);
}


/* end '!=' with float*/
/* '!=' with int*/

int checkNotEqualIntLiteralCompare1(void) {
  // FALSE NEGATIVE!
  return (5 != 5);
  // C HECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkNotEqualIntLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f != g); // no warning
}

int checkNotEqualIntDeclCompare3(void) {
  int f = 7;
  return (f != 7); // no warning
}

int checkNotEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 != f); // no warning
}

int checkNotEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f != (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 != f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 != g + 4); // no warning
}

int checkNotEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (int)f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (char)f + 4);  // no warning
  return (0);
}
int checkNotEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t != u + t);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) != ((int)f + (3 - u)*t));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*(t + 1 != t + 1)));
  // CHECK-MESSAGES: :[[@LINE-1]]:59: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/*   end '!=' int          */
/* '!=' with int pointer */

int checkNotEqualIntPointerLiteralCompare1(void) {
  int* p = 0;
  return (p != 0); // no warning
}

int checkNotEqualIntPointerLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntPointerDeclCompare1(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f != g); // no warning
}

int checkNotEqualCastIntPointerDeclCompare11(void) {
  int k = 7;
  int* f = &k;
  return ((int*)f != (int*)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastIntPointerDeclCompare12(void) {
  int k = 7;
  int* f = &k;
  return ((int*)((char*)f) != (int*)f); // no warning
}
int checkNotEqualBinaryOpIntPointerCompare1(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = (f + 4 != f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare2(void) {
  int k = 7;
  int* f = &k;
  int* g = &k;
  return (f + 4 != g + 4); // no warning
}


int checkNotEqualBinaryOpIntPointerCompare3(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare4(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)((char*)f) + 4);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare1(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = ((f + (3)*t) != (f + (3)*t));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare2(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = (((3)*t + f) != (f + (3)*t));  // no warning
  return (0);
}
/*   end '!=' int*          */

/* '!=' with function*/

int checkNotEqualSameFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() != a+func());  // no warning
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualDifferentFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() != a+func2());  // no warning
  return (0);
}

int checkNotEqualSameFunctionSameParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) != a+funcParam(a));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualSameFunctionDifferentParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) != a+funcParam(b));  // no warning
  return (0);
}

/*   end '!=' with function*/

/*   end '!=' */



/* EQ operator           */

int checkEqualIntPointerDeclCompare(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f == g); // no warning
}

int checkEqualIntPointerDeclCompare0(void) {
  int k = 3;
  int* f = &k;
  return (f+1 == f+1);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: both sides of operator are equivalent [misc-redundant-expression]
}

/* EQ with float*/

int checkEqualFloatLiteralCompare1(void) {
  return (5.14F == 5.14F); // no warning
}

int checkEqualFloatLiteralCompare2(void) {
  return (6.14F == 7.14F); // no warning
}

int checkEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f == g); // no warning
}

int checkEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f == f); // no warning
}


int checkEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f == 7.1F); // no warning
}

int checkEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F == f); // no warning
}

int checkEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t == f); // no warning
}

int checkEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f == t); // no warning
}




int checkEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f == (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f == (int)f); // no warning
}
int checkEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F == f + 3.14F);  // no warning
  return (0);
}
int checkEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F == g + 3.14F); // no warning
}
int checkEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (int)f + 3.14F);  // no warning
  return (0);
}
int checkEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (char)f + 3.14F);  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) == ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*(f + t == f + t)));  // no warning
  return (0);
}





/* Equal with int*/

int checkEqualIntLiteralCompare1(void) {
  // FALSE NEGATIVE!
  return (5 == 5);
  // C HECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkEqualIntLiteralCompare2(void) {
  return (6 == 7); // no warning
}

int checkEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f == g); // no warning
}

int checkEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f == (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f == (int)f); // no warning
}

int checkEqualIntDeclCompare3(void) {
  int f = 7;
  return (f == 7); // no warning
}

int checkEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 == f); // no warning
}

int checkEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 == f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 == g + 4); // no warning
}


int checkEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (int)f + 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);

}
int checkEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (char)f + 4);  // no warning
  return (0);
}
int checkEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t == u + t);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) == ((int)f + (3 - u)*t));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*(t + 1 == t + 1)));
  // CHECK-MESSAGES: :[[@LINE-1]]:59: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* '==' with function*/

int checkEqualSameFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() == a+func());  // no warning
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualDifferentFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() == a+func2());  // no warning
  return (0);
}

int checkEqualSameFunctionSameParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) == a+funcParam(a));  // no warning
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualSameFunctionDifferentParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) == a+funcParam(b));  // no warning
  return (0);
}

/*   end '==' with function*/

/*   end EQ int          */

/* end EQ */


/*  LT */

/*  LT with float */

int checkLessThanFloatLiteralCompare1(void) {
  // FALSE NEGATIVE
  return (5.14F < 5.14F);
  // C HECK-MESSAGES: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanFloatLiteralCompare2(void) {
  // FALSE NEGATIVE
  return (6.14F < 7.14F); // no warning
  // C HECK-MESSAGES: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f < g); // no warning
}

int checkLessThanFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f < f);
  // C HECK-MESSAGES: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f < 7.1F); // no warning
}

int checkLessThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F < f); // no warning
}

int checkLessThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f < t); // no warning
}


int checkLessThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f < (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkLessThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F < f + 3.14F);  // no warning
  return (0);
}
int checkLessThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F < g + 3.14F); // no warning
}
int checkLessThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (int)f + 3.14F);  // no warning
  return (0);
}
int checkLessThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (char)f + 3.14F);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) < ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*(f + t < f + t)));  // no warning
  return (0);
}

/*  end LT with float */

/*  LT with int */


int checkLessThanIntLiteralCompare1(void) {
  // FALSE NEGATIVE!
  return (5 < 5);
  // C HECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanIntLiteralCompare2(void) {
  return (6 < 7); // no warning
}

int checkLessThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f < g); // no warning
}

int checkLessThanIntDeclCompare3(void) {
  int f = 7;
  return (f < 7); // no warning
}

int checkLessThanIntDeclCompare4(void) {
  int f = 7;
  return (7 < f); // no warning
}

int checkLessThanIntDeclCompare5(void) {
  int f = 7;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanIntDeclCompare6(void) {
  int f = 7;
  int t = 7;
  return (f < t); // no warning
}

int checkLessThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f < (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkLessThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 < f + 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 < g + 3); // no warning
}
int checkLessThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (int)f + 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (char)f + 3);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) < ((int)f + (3 - u)*t));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*(t + u < t + u)));
  // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* end LT with int */

/* end LT */


/* GT */

/* GT with float */

int checkGreaterThanFloatLiteralCompare1(void) {
  // FALSE NEGATIVE!
  return (5.14F > 5.14F);
  // C HECK-MESSAGES: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkGreaterThanFloatLiteralCompare2(void) {
  return (6.14F > 7.14F); // no warning
}

int checkGreaterThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;

  return (f > g); // no warning
}

int checkGreaterThanFloatDeclCompare12(void) {
  float f = 7.1F;
  // FALSE NEGATIVE!
  return (f > f);
  // C HECK-MESSAGES: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}


int checkGreaterThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f > 7.1F); // no warning
}

int checkGreaterThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F > f); // no warning
}

int checkGreaterThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t > f); // no warning
}

int checkGreaterThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f > t); // no warning
}

int checkGreaterThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f > (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkGreaterThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F > f + 3.14F);  // no warning
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F > g + 3.14F); // no warning
}
int checkGreaterThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (int)f + 3.14F);  // no warning
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (char)f + 3.14F);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) > ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  // FALSE NEGATIVE!
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*(f + t > f + t)));  // no warning
  // C HECK-MESSAGES: :[[@LINE-1]]:66: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/*  end GT with float */

/*  GT with int */


int checkGreaterThanIntLiteralCompare1(void) {
  // FALSE NEGATIVE!
  return (5 > 5);
  // C HECK-MESSAGES: :[[@LINE-1]]:66: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkGreaterThanIntLiteralCompare2(void) {
  return (6 > 7); // no warning
}

int checkGreaterThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;

  return (f > g); // no warning
}

int checkGreaterThanIntDeclCompare3(void) {
  int f = 7;
  return (f > 7); // no warning
}

int checkGreaterThanIntDeclCompare4(void) {
  int f = 7;
  return (7 > f); // no warning
}

int checkGreaterThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f > (int)f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkGreaterThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 > f + 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 > g + 3); // no warning
}
int checkGreaterThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (int)f + 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (char)f + 3);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) > ((int)f + (3 - u)*t));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*(t + u > t + u)));
  // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* end GT with int */

/* end GT */

void test_identical_bitwise1() {
  // FALSE NEGATIVE!
  int a = 5 | 5; // expected-warning {{identical expressions on both sides of bitwise operator}}
  // C HECK-MESSAGES: :[[@LINE-3]]:44: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}

void test_identical_bitwise2() {
  int a = 5;
  int b = a | a; // expected-warning {{identical expressions on both sides of bitwise operator}}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

void test_identical_bitwise3() {
  int a = 5;
  int b = (a | a); // expected-warning {{identical expressions on both sides of bitwise operator}}
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

void test_identical_bitwise4() {
  int a = 4;
  int b = a | 4; // no-warning
}

void test_identical_bitwise5() {
  int a = 4;
  int b = 4;
  int c = a | b; // no-warning
}

void test_identical_bitwise6() {
  int a = 5;
  int b = a | 4 | a; // expected-warning {{identical expressions on both sides of bitwise operator}}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: operator has equivalent nested operands [misc-redundant-expression] 
}

void test_identical_bitwise7() {
  int a = 5;
  int b = func() | func(); // no-warning
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

void test_identical_logical1(int a) {
  if (a == 4 && a == 4) // expected-warning {{identical expressions on both sides of logical operator}}
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

void test_identical_logical2(int a) {
  if (a == 4 || a == 5 || a == 4) // expected-warning {{identical expressions on both sides of logical operator}}
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:24: warning: operator has equivalent nested operands [misc-redundant-expression] 
}

void test_identical_logical3(int a) {
  if (a == 4 || a == 5 || a == 6) // no-warning
    ;
}

void test_identical_logical4(int a) {
  if (a == func() || a == func()) // no-warning
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:19: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
void test_identical_logical5(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 6) // no-warning
    ;
}

void test_identical_logical6(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 5) // expected-warning {{identical expressions on both sides of logical operator}}
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:24: warning: both sides of operator are equivalent [misc-redundant-expression] 
}

void test_identical_logical7(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 && y == 5 || x == 4)
    ;
}

void test_identical_logical8(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 || y == 5 && x == 4)
    ;
}

void test_identical_logical9(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 || x == 4 && y == 5)
    ;
}
#pragma clang diagnostic pop

unsigned test_unsigned(unsigned a) {
  unsigned b = 1;
  a = a > 5 ? b : b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
  return a;
}

void test_signed() {
  int a = 0;
  a = a > 5 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_bool(bool a) {
  a = a > 0 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_float() {
  float a = 0;
  float b = 0;
  a = a > 5 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

const char *test_string() {
  float a = 0;
  return a > 5 ? "abc" : "abc";
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:16: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_unsigned_expr() {
  unsigned a = 0;
  unsigned b = 0;
  a = a > 5 ? a+b : a+b;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_signed_expr() {
  int a = 0;
  int b = 1;
  a = a > 5 ? a+b : a+b;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_bool_expr(bool a) {
  bool b = 0;
  a = a > 0 ? a&&b : a&&b;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
  // CHECK-CLONE: :[[@LINE-2]]:13: warning: conditional operator with identical true and false expressions [bugprone-branch-clone] 
}

void test_unsigned_expr_negative() {
  unsigned a = 0;
  unsigned b = 0;
  a = a > 5 ? a+b : b+a; // no warning
}

void test_signed_expr_negative() {
  int a = 0;
  int b = 1;
  a = a > 5 ? b+a : a+b; // no warning
}

void test_bool_expr_negative(bool a) {
  bool b = 0;
  a = a > 0 ? a&&b : b&&a; // no warning
}

void test_float_expr_positive() {
  float a = 0;
  float b = 0;
  a = a > 5 ? a+b : a+b; 
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}

void test_expr_positive_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func();
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}

void test_expr_negative_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func2(); // no warning
}

void test_expr_positive_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(b) : a+funcParam(b);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}

void test_expr_negative_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(a) : a+funcParam(b); // no warning
}

void test_expr_positive_inc() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a++ : a++; // no warning
}

void test_expr_negative_inc() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a++ : b++; // no warning
}

void test_expr_positive_assign() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a=1 : a=1; // no warning
}

void test_expr_negative_assign() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a=1 : a=2; // no warning
}

void test_signed_nested_expr() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? a+b+(c+a)*(a + b*(c+a)) : a+b+(c+a)*(a + b*(c+a));
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}

void test_signed_nested_expr_negative() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? a+b+(c+a)*(a + b*(c+a)) : a+b+(c+a)*(a + b*(a+c)); // no warning
}

void test_signed_nested_cond_expr_negative() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? (b > 5 ? 1 : 4) : (b > 5 ? 2 : 4); // no warning
}

void test_signed_nested_cond_expr() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? (b > 5 ? 1 : 4) : (b > 5 ? 4 : 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression] 
}
