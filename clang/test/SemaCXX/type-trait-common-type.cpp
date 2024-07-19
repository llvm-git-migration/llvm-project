// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

#if !__has_builtin(__common_type)
#  error
#endif

// expected-note@*:* {{template declaration from hidden source: template <class, template <class> class, class>}}

void test() {
  __common_type<> a; // expected-error {{too few template arguments for template '__common_type'}}
}

struct empty_type {};

template <class T>
struct type_identity {
  using type = T;
};

template <class...>
struct common_type;

template <class... Args>
using common_type_base = __common_type<common_type, type_identity, empty_type, Args...>; // expected-error {{incomplete type 'common_type<Incomplete, Incomplete>' where a complete type is required}}

template <class... Args>
struct common_type : common_type_base<Args...> {};

struct Incomplete;

template<>
struct common_type<Incomplete, Incomplete>; // expected-note {{forward declaration}}

static_assert(__is_same(common_type_base<>, empty_type));
static_assert(__is_same(common_type_base<Incomplete>, empty_type)); // expected-note {{requested here}}
static_assert(__is_same(common_type_base<char>, type_identity<char>));
static_assert(__is_same(common_type_base<int>, type_identity<int>));
static_assert(__is_same(common_type_base<const int>, type_identity<int>));
static_assert(__is_same(common_type_base<volatile int>, type_identity<int>));
static_assert(__is_same(common_type_base<const volatile int>, type_identity<int>));
static_assert(__is_same(common_type_base<int[]>, type_identity<int*>));
static_assert(__is_same(common_type_base<const int[]>, type_identity<const int*>));
static_assert(__is_same(common_type_base<void(&)()>, type_identity<void(*)()>));
static_assert(__is_same(common_type_base<int[], int[]>, type_identity<int*>));

static_assert(__is_same(common_type_base<int, int>, type_identity<int>));
static_assert(__is_same(common_type_base<int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<long, int>, type_identity<long>));
static_assert(__is_same(common_type_base<long, long>, type_identity<long>));

static_assert(__is_same(common_type_base<const int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<const volatile int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<int, const long>, type_identity<long>));
static_assert(__is_same(common_type_base<int, const volatile long>, type_identity<long>));

static_assert(__is_same(common_type_base<int*, long*>, empty_type));

static_assert(__is_same(common_type_base<int, long, float>, type_identity<float>));
static_assert(__is_same(common_type_base<unsigned, char, long>, type_identity<long>));

struct NoCommonType {};

template <>
struct common_type<NoCommonType, NoCommonType> {};

struct CommonTypeInt {};

template <>
struct common_type<CommonTypeInt, CommonTypeInt> {
  using type = int;
};

template <>
struct common_type<CommonTypeInt, int> {
  using type = int;
};

template <>
struct common_type<int, CommonTypeInt> {
  using type = int;
};

static_assert(__is_same(common_type_base<NoCommonType>, empty_type));
static_assert(__is_same(common_type_base<CommonTypeInt>, type_identity<int>));
static_assert(__is_same(common_type_base<NoCommonType, NoCommonType, NoCommonType>, empty_type));
static_assert(__is_same(common_type_base<CommonTypeInt, CommonTypeInt, CommonTypeInt>, type_identity<int>));
static_assert(__is_same(common_type_base<CommonTypeInt&, CommonTypeInt&&>, type_identity<int>));

static_assert(__is_same(common_type_base<void, int>, empty_type));
static_assert(__is_same(common_type_base<void, void>, type_identity<void>));
static_assert(__is_same(common_type_base<const void, void>, type_identity<void>));
static_assert(__is_same(common_type_base<void, const void>, type_identity<void>));

template <class T>
struct ConvertibleTo {
  operator T();
};

static_assert(__is_same(common_type_base<ConvertibleTo<int>>, type_identity<ConvertibleTo<int>>));
static_assert(__is_same(common_type_base<ConvertibleTo<int>, int>, type_identity<int>));
static_assert(__is_same(common_type_base<ConvertibleTo<int&>, ConvertibleTo<long&>>, type_identity<long>));

struct ConvertibleToB;

struct ConvertibleToA {
  operator ConvertibleToB();
};

struct ConvertibleToB {
  operator ConvertibleToA();
};

static_assert(__is_same(common_type_base<ConvertibleToA, ConvertibleToB>, empty_type));

struct const_ref_convertible {
  operator int&() const &;
  operator int&() && = delete;
};

#if __cplusplus >= 202002L
static_assert(__is_same(common_type_base<const_ref_convertible, int &>, type_identity<int>));
#else
static_assert(__is_same(common_type_base<const_ref_convertible, int &>, empty_type));
#endif
