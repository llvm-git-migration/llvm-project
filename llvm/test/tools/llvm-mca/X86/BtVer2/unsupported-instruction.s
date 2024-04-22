# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions %s |& FileCheck --check-prefix=CHECK-SKIP %s
# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s |& FileCheck --check-prefix=CHECK-ERROR %s

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions to ignore.

# Currently lacks scheduling information and is therefore reported as unsupported by llvm-mca.
# This may change some day in which case an alternative unsupported input would need to be found or the test removed.
bzhi %eax, %ebx, %ecx

# Supported instruction that may be analysed.
add %eax, %eax

# CHECK-SKIP: Iterations:        100
# CHECK-SKIP: Instructions:      100
# CHECK-SKIP: Total Cycles:      103
# CHECK-SKIP: Total uOps:        100

# CHECK-SKIP: Dispatch Width:    2
# CHECK-SKIP: uOps Per Cycle:    0.97
# CHECK-SKIP: IPC:               0.97
# CHECK-SKIP: Block RThroughput: 0.5