# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-zicfiss - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental < %s \
# RUN:     | llvm-objdump -d --mattr=+experimental-zicfiss - \
# RUN:     | FileCheck -check-prefix=CHECK-INST-ALIAS %s
#
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK-INVALID %s

##################################
# Experimental User CSRs
##################################

.option push
.option arch, +zicfiss
# ssp
# name
# CHECK-INST: csrrs t1, ssp, zero
# CHECK-ENC:  encoding: [0x73,0x23,0x10,0x01]
# CHECK-INST-ALIAS: csrr t1, ssp
# uimm12
# CHECK-INST: csrrs t2, ssp, zero
# CHECK-ENC:  encoding: [0xf3,0x23,0x10,0x01]
# CHECK-INST-ALIAS: csrr t2, ssp
# name
# CHECK-INVALID: error: unexpected experimental extensions
csrrs t1, ssp, zero # CHECK-INVALID: :[[@LINE]]:11: error: system register 'ssp' requires 'experimental-zicfiss' to be enabled
# uimm12
csrrs t2, 0x011, zero
.option pop
