# RUN: llvm-mc -filetype obj -triple amd64-openbsd %s | llvm-readobj -hS - | FileCheck %s
# CHECK: OS/ABI: OpenBSD
