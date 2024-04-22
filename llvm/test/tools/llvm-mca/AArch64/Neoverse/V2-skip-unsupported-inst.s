# RUN: llvm-mca -march=aarch64 -mcpu=neoverse-v2 -skip-unsupported-instructions %s |& FileCheck --check-prefix=CHECK-SKIP %s
# RUN: not llvm-mca -march=aarch64 -mcpu=neoverse-v2 %s |& FileCheck --check-prefix=CHECK-ERROR %s

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions to ignore.

# Currently lacks scheduling information and is therefore reported as unsupported by llvm-mca.
# This may change some day in which case an alternative unsupported input would need to be found or the test removed.
steor x0, [x1]
