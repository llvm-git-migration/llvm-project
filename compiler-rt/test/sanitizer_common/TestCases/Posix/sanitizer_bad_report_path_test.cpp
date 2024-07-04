// Test __sanitizer_set_report_path and __sanitizer_get_report_path with an
// unwritable directory.
// RUN: rm -rf %t.report_path && mkdir -p %t.report_path
// RUN: chmod u-w %t.report_path || true
// RUN: %clangxx -O2 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=FAIL

// The chmod is not working on the android bot for some reason.
// UNSUPPORTED: android

#include <assert.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
#  include <linux/capability.h>

/* Use capget() and capset() from glibc. */
extern "C" int capget(cap_user_header_t header, cap_user_data_t data);
extern "C" int capset(cap_user_header_t header, const cap_user_data_t data);

static void try_drop_cap_dac_override(void) {
  struct __user_cap_header_struct hdr = {
      .version = _LINUX_CAPABILITY_VERSION_1,
      .pid = 0,
  };
  struct __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_1];
  if (!capget(&hdr, data)) {
    data[CAP_DAC_OVERRIDE >> 5].effective &= ~(1 << (CAP_DAC_OVERRIDE & 31));
    capset(&hdr, data);
  }
}
#else
static void try_drop_cap_dac_override(void) {}
#endif

volatile int *null = 0;

int main(int argc, char **argv) {
  try_drop_cap_dac_override();
  char buff[1000];
  sprintf(buff, "%s.report_path/report", argv[0]);
  __sanitizer_set_report_path(buff);
  assert(strncmp(buff, __sanitizer_get_report_path(), strlen(buff)) == 0);
  printf("Path %s\n", __sanitizer_get_report_path());
}

// FAIL: ERROR: Can't open file: {{.*}}Posix/Output/sanitizer_bad_report_path_test.cpp.tmp.report_path/report.
// FAIL-NOT: Path {{.*}}Posix/Output/sanitizer_bad_report_path_test.cpp.tmp.report_path/report.
