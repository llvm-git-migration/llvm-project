// Check that the module build daemon can create a unix socket

// RUN: rm -rf mbd-launch %t

// timeout should exit with status 124 which is treated as a failure on 
// windows. Ideally we would be like to check the exit code and only return true
// if it equals 124 but global bash sysmbols like $? are not surported by lit

// RUN: timeout --signal=SIGTERM 2 %clang -cc1modbuildd mbd-launch -v || true
// RUN: cat mbd-launch/mbd.out | sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK: mbd created and binded to socket at: mbd-launch/mbd.sock

// Make sure socket file is removed when daemon exits
// [ ! -f "mbd-launch/mbd.socker" ]

// Make sure mbd.err is empty
// RUN: [ ! -s "mbd-launch/mbd.err" ]
