// RUN: %clang_cc1 -std=c++23 -verify %s

namespace usage_invalid {
  // FIXME: Should we diagnose a void return type?
  void voidreturn(int &param [[clang::lifetimebound]]);

  int *not_class_member() [[clang::lifetimebound]]; // expected-error {{non-member function has no implicit object parameter}}
  struct A {
    A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a constructor}}
    ~A() [[clang::lifetimebound]]; // expected-error {{cannot be applied to a destructor}}
    static int *static_class_member() [[clang::lifetimebound]]; // expected-error {{static member function has no implicit object parameter}}
    int *explicit_object(this A&) [[clang::lifetimebound]]; // expected-error {{explicit object member function has no implicit object parameter}}
    int not_function [[clang::lifetimebound]]; // expected-error {{only applies to parameters and implicit object parameters}}
    int [[clang::lifetimebound]] also_not_function; // expected-error {{cannot be applied to types}}
  };
  int *attr_with_param(int &param [[clang::lifetimebound(42)]]); // expected-error {{takes no arguments}}
}

namespace usage_ok {
  struct IntRef { int *target; };

  int &refparam(int &param [[clang::lifetimebound]]);
  int &classparam(IntRef param [[clang::lifetimebound]]);

  // Do not diagnose non-void return types; they can still be lifetime-bound.
  long long ptrintcast(int &param [[clang::lifetimebound]]) {
    return (long long)&param;
  }
  // Likewise.
  int &intptrcast(long long param [[clang::lifetimebound]]) {
    return *(int*)param;
  }

  struct A {
    A();
    A(int);
    int *class_member() [[clang::lifetimebound]];
    operator int*() [[clang::lifetimebound]];
  };

  int *p = A().class_member(); // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  int *q = A(); // expected-warning {{temporary whose address is used as value of local variable 'q' will be destroyed at the end of the full-expression}}
  int *r = A(1); // expected-warning {{temporary whose address is used as value of local variable 'r' will be destroyed at the end of the full-expression}}

  void test_assignment() {
    p = A().class_member(); // expected-warning {{object backing the pointer p will be destroyed at the end of the full-expression}}
    p = {A().class_member()}; // expected-warning {{object backing the pointer p will be destroyed at the end of the full-expression}}
    q = A(); // expected-warning {{object backing the pointer q will be destroyed at the end of the full-expression}}
    r = A(1); // expected-warning {{object backing the pointer r will be destroyed at the end of the full-expression}}
  }

  struct FieldCheck {
    struct Set {
      int a;
    };
    struct Pair {
      const int& a;
      int b;
      Set c;
      int * d;
    };
    Pair p;  
    FieldCheck(const int& a): p(a){}
    Pair& getR() [[clang::lifetimebound]] { return p; }
    Pair* getP() [[clang::lifetimebound]] { return &p; }
    Pair* getNoLB() { return &p; }
  };
  void test_field_access() {
    int x = 0;
    const int& a = FieldCheck{x}.getR().a;
    const int& b = FieldCheck{x}.getP()->b;   // expected-warning {{temporary bound to local reference 'b' will be destroyed at the end of the full-expression}}
    const int& c = FieldCheck{x}.getP()->c.a; // expected-warning {{temporary bound to local reference 'c' will be destroyed at the end of the full-expression}}
    const int& d = FieldCheck{x}.getNoLB()->c.a;
    const int* e = FieldCheck{x}.getR().d;
  }
}

# 1 "<std>" 1 3
namespace std {
  using size_t = __SIZE_TYPE__;
  struct string {
    string();
    string(const char*);

    char &operator[](size_t) const [[clang::lifetimebound]];
  };
  string operator""s(const char *, size_t);

  struct string_view {
    string_view();
    string_view(const char *p [[clang::lifetimebound]]);
    string_view(const string &s [[clang::lifetimebound]]);
  };
  string_view operator""sv(const char *, size_t);

  struct vector {
    int *data();
    size_t size();
  };

  template<typename K, typename V> struct map {};
}
# 68 "attr-lifetimebound.cpp" 2

using std::operator""s;
using std::operator""sv;

namespace p0936r0_examples {
  std::string_view s = "foo"s; // expected-warning {{temporary}}

  std::string operator+(std::string_view s1, std::string_view s2);
  void f() {
    std::string_view sv = "hi";
    std::string_view sv2 = sv + sv; // expected-warning {{temporary}}
    sv2 = sv + sv; // FIXME: can we infer that we should warn here too?
  }

  struct X { int a, b; };
  const int &f(const X &x [[clang::lifetimebound]]) { return x.a; }
  const int &r = f(X()); // expected-warning {{temporary}}

  char &c = std::string("hello my pretty long strong")[0]; // expected-warning {{temporary}}

  struct reversed_range {
    int *begin();
    int *end();
    int *p;
    std::size_t n;
  };
  template <typename R> reversed_range reversed(R &&r [[clang::lifetimebound]]) {
    return reversed_range{r.data(), r.size()};
  }

  std::vector make_vector();
  void use_reversed_range() {
    // FIXME: Don't expose the name of the internal range variable.
    for (auto x : reversed(make_vector())) {} // expected-warning {{temporary implicitly bound to local reference will be destroyed at the end of the full-expression}}
  }

  template <typename K, typename V>
  const V &findOrDefault(const std::map<K, V> &m [[clang::lifetimebound]],
                         const K &key,
                         const V &defvalue [[clang::lifetimebound]]);

  // FIXME: Maybe weaken the wording here: "local reference 'v' could bind to temporary that will be destroyed at end of full-expression"?
  std::map<std::string, std::string> m;
  const std::string &v = findOrDefault(m, "foo"s, "bar"s); // expected-warning {{temporary bound to local reference 'v'}}
}

// definitions for std::move, std::forward et al.
namespace std {
inline namespace foo {

template <class T> struct remove_reference {
    typedef T type;
};
template <class T> struct remove_reference<T &> {
    typedef T type;
};
template <class T> struct remove_reference<T &&> {
    typedef T type;
};

template <class T> constexpr typename remove_reference<T>::type &&move(T &&t) {
    return static_cast<typename remove_reference<T>::type>(t);
}

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &t) {
    return static_cast<T &&>(t);
}

template <class T>
constexpr T &&forward(typename remove_reference<T>::type &&t) {
    return static_cast<T &&>(t);
}

template <class T> constexpr const T &as_const(T &x) { return x; }

template <class T, bool RValueRef> struct PickRef {
    using type = typename remove_reference<T>::type &;
};
template <class T> struct PickRef<T, true> {
    using type = typename remove_reference<T>::type &&;
};

template <class T> struct is_lvalue_reference {
    static constexpr bool value = false;
};

template <class T> struct is_lvalue_reference<T &> {
    static constexpr bool value = true;
};

template <class T> struct is_const {
    static constexpr bool value = false;
};

template <class T> struct is_const<const T> {
    static constexpr bool value = true;
};

template <bool B, class T, class F> struct conditional {
    using type = T;
};

template <class T, class F> struct conditional<false, T, F> {
    using type = F;
};

template <class U, class T>
using CopyConst = typename conditional<is_const<remove_reference<U>>::value,
                                       const T, T>::type;

template <class U, class T>
using OverrideRef =
    typename conditional<is_lvalue_reference<U &&>::value,
                         typename remove_reference<T>::type &,
                         typename remove_reference<T>::type &&>::type;

template <class U, class T>
using ForwardLikeRetType = OverrideRef<U &&, CopyConst<U, T>>;

template <class U>
constexpr auto forward_like(auto &&t) -> ForwardLikeRetType<U, decltype(t)> {
    return static_cast<ForwardLikeRetType<U, decltype(t)>>(t);
}

template <class T>
auto move_if_noexcept(T &t) ->
    typename PickRef<T, noexcept(T(static_cast<T &&>(t)))>::type {
    return static_cast<
        typename PickRef<T, noexcept(T(static_cast<T &&>(t)))>::type>(t);
}

template <class T> T *addressof(T &arg) {
    return reinterpret_cast<T *>(
        &const_cast<char &>(reinterpret_cast<const volatile char &>(arg)));
}

template<typename T>
struct basic_string_view {
  basic_string_view(const T *);
};

template <class T> struct span {
  template<size_t _ArrayExtent>
	span(const T (&__arr)[_ArrayExtent]) noexcept;
};

} // namespace foo
} // namespace std

namespace move_forward_et_al_examples {
  struct S {
    S &self() [[clang::lifetimebound]] { return *this; }
  };

  S &&Move = std::move(S{}); // expected-warning {{temporary bound to local reference 'Move' will be destroyed at the end of the full-expression}}
  S MoveOk = std::move(S{});

  S &&Forward = std::forward<S &&>(S{}); // expected-warning {{temporary bound to local reference 'Forward' will be destroyed at the end of the full-expression}}
  S ForwardOk = std::forward<S &&>(S{});

  S &&ForwardLike = std::forward_like<int&&>(S{}); // expected-warning {{temporary bound to local reference 'ForwardLike' will be destroyed at the end of the full-expression}}
  S ForwardLikeOk = std::forward_like<int&&>(S{});

  const S &Const = std::as_const(S{}.self()); // expected-warning {{temporary bound to local reference 'Const' will be destroyed at the end of the full-expression}}
  const S ConstOk = std::as_const(S{}.self());

  S &&MoveIfNoExcept = std::move_if_noexcept(S{}.self()); // expected-warning {{temporary bound to local reference 'MoveIfNoExcept' will be destroyed at the end of the full-expression}}
  S MoveIfNoExceptOk = std::move_if_noexcept(S{}.self());

  S *AddressOf = std::addressof(S{}.self()); // expected-warning {{temporary whose address is used as value of local variable 'AddressOf' will be destroyed at the end of the full-expression}}
  S X;
  S *AddressOfOk = std::addressof(X);
} // namespace move_forward_et_al_examples

namespace ctor_cases {
std::basic_string_view<char> test1() {
  char abc[10];
  return abc;  // expected-warning {{address of stack memory associated with local variable}}
}

std::span<int> test2() {
  int abc[10];
  return abc; // expected-warning {{address of stack memory associated with local variable}}
}
} // namespace ctor_cases

namespace GH106372 {
class [[gsl::Owner]] Foo {};
class [[gsl::Pointer]] FooView {};

class NonAnnotatedFoo {};
class NonAnnotatedFooView {};

template <typename T>
struct StatusOr {
  template <typename U = T>
  StatusOr& operator=(U&& v [[clang::lifetimebound]]);
};

void test(StatusOr<FooView> foo1, StatusOr<NonAnnotatedFooView> foo2) {
  foo1 = Foo(); // expected-warning {{object backing the pointer foo1 will be destroyed at the end}}
  // No warning on non-gsl annotated types.
  foo2 = NonAnnotatedFoo();
}
} // namespace GH106372

namespace lifetime_capture_by {

struct S {
  const int *x;
  void captureInt(const int&x [[clang::lifetime_capture_by(this)]]) { this->x = &x; }
  void captureSV(std::string_view sv [[clang::lifetime_capture_by(this)]]);
};

///////////////////////////
// Test for valid usages.
///////////////////////////
[[clang::lifetime_capture_by(unknown)]] // expected-error {{'lifetime_capture_by' attribute only applies to parameters and implicit object parameters}}
void nonMember(
    const int &x1 [[clang::lifetime_capture_by(s, t)]],
    S &s,
    S &t,
    const int &x2 [[clang::lifetime_capture_by(12345 + 12)]], // expected-error {{'lifetime_capture_by' attribute argument 12345 + 12 is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x3 [[clang::lifetime_capture_by(abcdefgh)]],   // expected-error {{'lifetime_capture_by' attribute argument 'abcdefgh' is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x4 [[clang::lifetime_capture_by("abcdefgh")]], // expected-error {{'lifetime_capture_by' attribute argument "abcdefgh" is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x5 [[clang::lifetime_capture_by(this)]], // expected-error {{'lifetime_capture_by' argument references unavailable implicit 'this'}}
    const int &x6 [[clang::lifetime_capture_by()]], // expected-error {{'lifetime_capture_by' attribute specifies no capturing entity}}
    const int& x7 [[clang::lifetime_capture_by(u, 
                                               x7)]], // expected-error {{'lifetime_capture_by' argument references itself}}
    const S& u
  )
{
  s.captureInt(x1);
}

struct T {
  void member(
    const int &x [[clang::lifetime_capture_by(s)]], 
    S &s,
    S &t,            
    const int &y [[clang::lifetime_capture_by(s)]],
    const int &z [[clang::lifetime_capture_by(this, x, y)]],
    const int &u [[clang::lifetime_capture_by(global, x, s)]])
  {
    s.captureInt(x);
  }
};

struct ThisIsCaptured {
  void foo(S& s) [[clang::lifetime_capture_by(s)]];
  void bar(S& s) [[clang::lifetime_capture_by(abcd)]]; // expected-error {{'lifetime_capture_by' attribute argument 'abcd' is not a known function parameter}}
  void baz(S& s) [[clang::lifetime_capture_by(this)]]; // expected-error {{'lifetime_capture_by' argument references itself}}
};

///////////////////////////
// Detect dangling cases.
///////////////////////////
void captureInt(const int&x [[clang::lifetime_capture_by(s)]], S&s);
void noCaptureInt(int x [[clang::lifetime_capture_by(s)]], S&s);

std::string_view substr(const std::string& s [[clang::lifetimebound]]);
std::string_view strcopy(const std::string& s);

void captureSV(std::string_view x [[clang::lifetime_capture_by(s)]], S&s);
void noCaptureSV(std::string_view x, S&s);

void use() {
  S s;
  int local;
  captureInt(1, // expected-warning {{object captured by the 's' will be destroyed at the end of the full-expression}}
            s);
  captureInt(local, s);
  
  noCaptureInt(1, s);
  noCaptureInt(local, s);

  std::string_view local_sv;
  captureSV(local_sv, s);
  captureSV(local_sv + local_sv, s);
  captureSV(std::string(), // expected-warning {{object captured by the 's'}}
            s);
  captureSV(substr(
      std::string() // expected-warning {{object captured by the 's'}}
      ), s);
  captureSV(strcopy(std::string()), s);
  
  noCaptureSV(local_sv, s);
  noCaptureSV(std::string(), s);
  noCaptureSV(substr(std::string()), s);

  s.captureInt(1); // expected-warning {{object captured by the 's' will be destroyed at the end of the full-expression}}
  s.captureSV(std::string()); // expected-warning {{object captured by the 's'}}
  s.captureSV(substr(std::string())); // expected-warning {{object captured by the 's'}}
  s.captureSV(strcopy(std::string()));
}
} // namespace lifetime_capture_by_usage

// Test for templated code.
// 2 nested function calls foo(sv, bar(sv, setsv));