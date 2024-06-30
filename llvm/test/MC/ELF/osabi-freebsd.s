# RUN: llvm-mc -filetype obj -triple amd64-freebsd %s | llvm-readobj -hS - | FileCheck %s
# CHECK: OS/ABI: FreeBSD
