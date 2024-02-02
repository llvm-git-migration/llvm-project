// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

extern void __tgt_rtl_init(void);
extern void __tgt_rtl_deinit(void);

// Sanity checks to make sure that this works and is thread safe.
int main() {
  __tgt_rtl_init();
#pragma omp parallel num_threads(8)
  {
    __tgt_rtl_init();
    __tgt_rtl_deinit();
  }
  __tgt_rtl_deinit();

  __tgt_rtl_init();
  __tgt_rtl_deinit();

  // CHECK: PASS
  printf("PASS\n");
}
