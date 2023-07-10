// Check that module build daemon can create unix socket

// REQUIRES: !system-windows

// RUN: if pgrep -f "cc1modbuildd mbd-launch"; then pkill -f "cc1modbuildd mbd-launch"; fi
// RUN: rm -rf mbd-launch %t

// RUN: %clang -cc1modbuildd mbd-launch -v
// RUN: cat mbd-launch/mbd.out | FileCheck %s

// CHECK: mbd created and binded to socket address at: mbd-launch/mbd.sock

// RUN: if pgrep -f "cc1modbuildd mbd-launch"; then pkill -f "cc1modbuildd mbd-launch"; fi
// RUN: rm -rf mbd-launch %t
