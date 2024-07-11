// RUN: %clangxx -O0 %s -o %t && %run %t
// RUN: %env_tool_opts=handle_segv=2 not %run %t 3 2>&1 | FileCheck --check-prefixes=CHECK1,CHECK

// UNSUPPORTED: android

#include <assert.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

// CHECK: [[SAN:.*Sanitizer]]:DEADLYSIGNAL
// CHECK: ERROR: [[SAN]]: SEGV on unknown address {{0x[^ ]*}} (pc
int main(void) {
  int fd = open("/proc/self/stat", O_RDONLY);
  char bufa[7];
  char bufb[7];
  struct iovec vec[2];
  vec[0].iov_base = bufa + 4;
  vec[0].iov_len = 1;
  vec[1].iov_base = bufb;
  vec[1].iov_len = sizeof(bufb);
  ssize_t rd = preadv2(fd, vec, 2, 0, 0);
  assert(rd > 0);
  vec[0].iov_base = bufa;
  rd = preadv2(fd, vec, 2, 0, 0);
  assert(rd > 0);
  vec[1].iov_len = 1024;
  preadv2(fd, vec, 2, 0, 0);
  // CHECK1: #{{[0-9]+ .*}}main {{.*}}preadv2.cpp:[[@LINE-1]]:[[TAB:[0-9]+]]
  // CHECK1: SUMMARY: [[SAN]]: SEGV {{.*}}preadv2.cpp:[[@LINE-2]]:[[TAB]] in main
  return 0;
}
