# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions %s |& FileCheck --check-prefixes=CHECK-ALL,CHECK-SKIP %s
# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s |& FileCheck --check-prefixes=CHECK-ALL,CHECK-ERROR %s

# Currently lacks scheduling information and is therefore reported as unsupported by llvm-mca.
# This may change some day in which case an alternative unsupported input would need to be found or the test removed.
bzhi %eax, %ebx, %ecx

# CHECK-ALL-NOT: error

# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions to ignore.

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-SKIP: note: instruction:      bzhil   %eax, %ebx, %ecx
# CHECK-SKIP: error: no assembly instructions found.
