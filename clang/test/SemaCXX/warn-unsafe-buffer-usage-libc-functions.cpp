// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s

typedef struct {} FILE;
void memcpy();
void __asan_memcpy();
void strcpy();
void strcpy_s();
void wcscpy_s();
unsigned strlen( const char* str );
int fprintf( FILE* stream, const char* format, ... );
int printf( const char* format, ... );
int sprintf( char* buffer, const char* format, ... );
int snprintf( char* buffer, unsigned buf_size, const char* format, ... );
int vsnprintf( char* buffer, unsigned buf_size, const char* format, ... );
int sscanf_s(const char * buffer, const char * format, ...);
int sscanf(const char * buffer, const char * format, ... );

namespace std {
  template< class InputIt, class OutputIt >
  OutputIt copy( InputIt first, InputIt last,
		 OutputIt d_first );

  struct iterator{};
  template<typename T>
  struct span {
    T * ptr;
    T * data();
    unsigned size_bytes();
    unsigned size();
    iterator begin() const noexcept;
    iterator end() const noexcept;
  };

  template<typename T>
  struct basic_string {
    T* p;
    T *c_str();
    T *data();
    unsigned size_bytes();
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;
}

void f(char * p, char * q, std::span<char> s) {
  memcpy();                   // expected-warning{{function introduces unsafe buffer manipulation}}
  __builtin_memcpy(p, q, 64); // expected-warning{{function introduces unsafe buffer manipulation}}
  __builtin___memcpy_chk(p, q, 8, 64);  // expected-warning{{function introduces unsafe buffer manipulation}}
  __asan_memcpy();                      // expected-warning{{function introduces unsafe buffer manipulation}}
  strcpy();                   // expected-warning{{function introduces unsafe buffer manipulation}}
  strcpy_s();                 // expected-warning{{function introduces unsafe buffer manipulation}}
  wcscpy_s();                 // expected-warning{{function introduces unsafe buffer manipulation}}


  /* Test printfs */

  fprintf((FILE*)p, "%s%d", p, *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{use 'std::string::c_str' as pointer to guarantee null-termination}}
  printf("%s%d", p, *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{use 'std::string::c_str' as pointer to guarantee null-termination}}
  sprintf(q, "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{change to 'snprintf' for explicit bounds checking}}
  snprintf(q, 10, "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{use 'std::string::c_str' as pointer to guarantee null-termination}}
  snprintf(s.data(), s.size(), "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{use 'std::string::c_str' as pointer to guarantee null-termination}}
  vsnprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}}
  sscanf(p, "%s%d", "hello", *p);  // expected-warning{{function introduces unsafe buffer manipulation}}
  sscanf_s(p, "%s%d", "hello", *p);  // expected-warning{{function introduces unsafe buffer manipulation}}
  fprintf((FILE*)p, "%s%d", "hello", *p); // no warn
  printf("%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // no warn
  strlen("hello");// no warn
}

void v(std::string s1, std::wstring s2) {
  snprintf(s1.data(), s1.size_bytes(), "%s%d", s1.c_str(), 0); // no warn
}


void g(char *begin, char *end, char *p, std::span<char> s) {
  std::copy(begin, end, p); // no warn
  std::copy(s.begin(), s.end(), s.begin()); // no warn
}
