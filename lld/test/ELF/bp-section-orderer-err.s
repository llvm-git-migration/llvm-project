# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
# RUN: echo "A B 5" > %t.call_graph
# RUN: echo "B C 50" >> %t.call_graph
# RUN: echo "C D 40" >> %t.call_graph
# RUN: echo "D B 10" >> %t.call_graph
# RUN: not ld.lld -o /dev/null %t --irpgo-profile %s --call-graph-ordering-file=%t.call_graph 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# RUN: not ld.lld -o /dev/null %t --irpgo-profile=%s --call-graph-ordering-file=%t.call_graph 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# IRPGO-ERR: --irpgo-profile is incompatible with --call-graph-ordering-file

# RUN: not ld.lld -o /dev/null --bp-compression-sort=function --call-graph-ordering-file %t.call_graph 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# COMPRESSION-ERR: --bp-compression-sort is incompatible with --call-graph-ordering-file

# RUN: not ld.lld -o /dev/null --bp-compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# COMPRESSION-MALFORM: unknown value 'malformed' for --bp-compression-sort=

# RUN: not ld.lld -o /dev/null --bp-compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=STARTUP
# STARTUP: --bp-compression-sort-startup-functions must be used with --irpgo-profile

# CHECK: B
# CHECK-NEXT: C
# CHECK-NEXT: D
# CHECK-NEXT: A

.section    .text.A,"ax",@progbits
.globl  A
A:
 nop

.section    .text.B,"ax",@progbits
.globl  B
B:
 nop

.section    .text.C,"ax",@progbits
.globl  C
C:
 nop

.section    .text.D,"ax",@progbits
.globl  D
D:
 nop
