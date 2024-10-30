// RUN: %check_clang_tidy %s bugprone-undefined-sprintf-overlap %t

using size_t = decltype(sizeof(int));

extern "C" int sprintf(char *s, const char *format, ...);
extern "C" int snprintf(char *s, size_t n, const char *format, ...);

namespace std {
  int snprintf(char *s, size_t n, const char *format, ...);
}

void first_arg_overlaps() {
  char buf[10];
  sprintf(buf, "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: argument 'buf' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: argument 'buf' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  std::snprintf(buf, sizeof(buf), "%s", buf);
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: argument 'buf' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  char* c = &buf[0];
  sprintf(c, "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: argument 'c' overlaps the first argument in 'sprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
  snprintf(c, sizeof(buf), "%s", c);
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: argument 'c' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]

  snprintf(c, sizeof(buf), "%s%s", c, c);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: argument 'c' overlaps the first argument in 'snprintf', which is undefined behavior [bugprone-undefined-sprintf-overlap]
}
