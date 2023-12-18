// Check that the module build daemon can create a unix socket

// REQUIRES: !system-windows
// RUN: rm -rf mbd-launch %t

// The module build daemon relies on a call to llvm::sys::ExecuteNoWait by the
// frontend where Detached == true to be detached from the terminal so when 
// using -cc1modbuildd directly the command needs to be killed manually

// RUN: timeout --preserve-status --signal=SIGTERM 2 %clang -cc1modbuildd mbd-launch -v
// RUN: cat mbd-launch/mbd.out | FileCheck %s

// CHECK: mbd created and binded to socket at: mbd-launch/mbd.sock
