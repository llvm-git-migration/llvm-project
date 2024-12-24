// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify

// This program causes clang 19 and earlier to crash because
// EnumDecl::PromotionType has not been set on the instantiated enum.
// See GitHub Issue #117960.
namespace Issue117960 {
template <typename T>
struct A {
  enum E : T;
};

int b = A<int>::E{} + 0;
}


namespace test {
template <typename T1, typename T2>
struct IsSame {
  static constexpr bool check() { return false; }
};

template <typename T>
struct IsSame<T, T> {
  static constexpr bool check() { return true; }
};
}  // namespace test


template <typename T>
struct S1 {
  enum E : T;
};
// checks if EnumDecl::PromotionType is set
int X1 = S1<int>::E{} + 0;
int Y1 = S1<unsigned>::E{} + 0;
static_assert(test::IsSame<decltype(S1<int>::E{}+0), int>::check(), "");
static_assert(test::IsSame<decltype(S1<unsigned>::E{}+0), unsigned>::check(), "");
char Z1 = S1<unsigned>::E(-1) + 0; // expected-warning{{implicit conversion from 'unsigned int' to 'char'}}

template <typename Traits>
struct S2 {
  enum E : typename Traits::IntegerType;
};

template <typename T>
struct Traits {
  typedef T IntegerType;
};

int X2 = S2<Traits<int>>::E{} + 0;
int Y2 = S2<Traits<unsigned>>::E{} + 0;
static_assert(test::IsSame<decltype(S2<Traits<int>>::E{}+0), int>::check(), "");
static_assert(test::IsSame<decltype(S2<Traits<unsigned>>::E{}+0), unsigned>::check(), "");


template <typename T>
struct S3 {
  enum E : unsigned;
};

int X3 = S3<float>::E{} + 0;

// fails in clang 19 and earlier (see the discussion on GitHub Issue #117960):
static_assert(test::IsSame<decltype(S3<float>::E{}+0), unsigned>::check(), "");

