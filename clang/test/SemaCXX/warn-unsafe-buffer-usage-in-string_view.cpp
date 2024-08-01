// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -verify %s -Wno-unsafe-buffer-usage-in-string-view -D_IGNORE_STRING_VIEW

namespace std {
  struct iterator{};

  template<typename T>
  struct basic_string_view {
    T* p;
    T *data();
    unsigned size_bytes();
    iterator begin();
    iterator end();

    constexpr basic_string_view( const T* s ){};
    constexpr basic_string_view( const T* s, unsigned count );
    constexpr basic_string_view( const basic_string_view& other ) noexcept = default;
    template< class It, class End >
    constexpr basic_string_view( It first, End last ) {};

  };

  typedef basic_string_view<char> string_view;
  typedef basic_string_view<wchar_t> wstring_view;

  template<typename T>
  struct basic_string {
    T *c_str();
    operator std::basic_string_view<T>() {
    }
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;

  template <class T> struct span {
    constexpr span(T *, unsigned){}
  };
}

void foo(std::string_view);
void f(char * p, std::string S, std::string_view V) {
  std::string_view SVs[] {S, "S"};
  foo(S);
  foo("S");
  foo({S});
  foo({"S"});
  foo(V);
#ifndef _IGNORE_STRING_VIEW
  foo(p);       // expected-warning{{construct string_view from raw pointers does not guarantee null-termination, construct from std::string instead}}
#else
  foo(p);       // no warn
#endif
#ifndef _IGNORE_STRING_VIEW
  foo({p, 10}); // expected-warning{{construct string_view from raw pointers does not guarantee null-termination, construct from std::string instead}}
#else
  foo({p, 10}); // no warn
#endif

  std::string_view SV{S};
  std::string_view SV2{"S"};
  std::string_view SV3 = S;
  std::string_view SV4 = "S";
  std::string_view SV5 = std::string_view(S);
#ifndef _IGNORE_STRING_VIEW
  std::string_view SV10 = p; // expected-warning{{construct string_view from raw pointers does not guarantee null-termination, construct from std::string instead}}
#else
  std::string_view SV10 = p; // no warn
#endif
#ifndef _IGNORE_STRING_VIEW
  std::string_view SV11{p};  // expected-warning{{construct string_view from raw pointers does not guarantee null-termination, construct from std::string instead}}
#else
  std::string_view SV11{p};  // no warn
#endif
#ifndef _IGNORE_STRING_VIEW
  std::string_view SV12{V.begin(), V.end()}; // expected-warning{{construct string_view from raw pointers does not guarantee null-termination, construct from std::string instead}}
#else
  std::string_view SV12{V.begin(), V.end()}; // no warn
#endif
}

void g(int * p) {
  // This warning is not affected:
  std::span<int> S{p, 10}; // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
}
