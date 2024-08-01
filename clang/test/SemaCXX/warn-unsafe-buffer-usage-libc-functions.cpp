// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -verify %s -Wno-unsafe-buffer-usage-in-string-view -D_IGNORE_STRING_VIEW

typedef struct {} FILE;
void memcpy();
void __builtin___memcpy_chk();
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

  template<typename T>
  struct basic_string_view {
    T* p;
    T *data();
    unsigned size_bytes();
  };

  typedef basic_string_view<char> string_view;
}

void f(char * p, char * q, std::span<char> s) {
  memcpy();                   // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  __builtin_memcpy(p, q, 64); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  __builtin___memcpy_chk();  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  strcpy();                   // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  strcpy_s();                 // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  wcscpy_s();                 // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}


  /* Test printfs */

  fprintf((FILE*)p, "%s%d", p, *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  printf("%s%d", p, *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  sprintf(q, "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  snprintf(q, 10, "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  snprintf(s.data(), s.size(), "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  vsnprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  sscanf(p, "%s%d", "hello", *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  sscanf_s(p, "%s%d", "hello", *p);  // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
  fprintf((FILE*)p, "%s%d", "hello", *p); // no warn
  printf("%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // no warn
  strlen("hello");// no warn
}

void v(std::string s, std::string_view sv) {
  snprintf(s.data(), s.size_bytes(), "%s%d", s.c_str(), 0); // no warn
#ifndef _IGNORE_STRING_VIEW
  snprintf(sv.data(), sv.size_bytes(), "%s%d", sv.data(), 0); // no warn
#else
  snprintf(sv.data(), sv.size_bytes(), "%s%d", sv.data(), 0); // expected-warning{{function introduces unsafe buffer manipulation}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#endif
}


void g(char *begin, char *end, char *p, std::span<char> s) {
  std::copy(begin, end, p); // no warn
  std::copy(s.begin(), s.end(), s.begin()); // no warn
}
