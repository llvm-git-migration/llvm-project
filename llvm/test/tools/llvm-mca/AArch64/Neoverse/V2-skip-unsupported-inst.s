# RUN: llvm-mca -march=aarch64 -mcpu=neoverse-v2 -skip-unsupported-instructions %s |& FileCheck --check-prefix=CHECK-SKIP %s
# RUN: not llvm-mca -march=aarch64 -mcpu=neoverse-v2 %s |& FileCheck --check-prefix=CHECK-ERROR %s

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions to ignore.

# Lacks scheduling information and is therefore unsupported by llvm-mca.
steor x0, [x1]
