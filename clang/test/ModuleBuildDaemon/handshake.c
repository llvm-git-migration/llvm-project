// Check that clang invocation can spawn and handshake with module build daemon

// REQUIRES: !system-windows

//  RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
//  RUN: rm -rf mbd-handshake %t
//  RUN: split-file %s %t

//--- main.c
int main() {return 0;}

// RUN: %clang -fmodule-build-daemon=mbd-handshake %t/main.c > output
// RUN: cat output | FileCheck %s

// CHECK: Completed successfull handshake with module build daemon

// RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
// RUN: rm -rf mbd-handshake %t
