// REQUIRES: !system-windows

//  RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
//  RUN: rm -rf mbd-handshake %t
//  RUN: split-file %s %t

//--- main.c
int main() {return 0;}

// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-new
// RUN: cat %t/output-new | FileCheck %s
// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-existing
// RUN: cat %t/output-existing | FileCheck %s --check-prefix=CHECK-EXIST

// COM: Check that clang invocation can spawn and handshake with module build daemon
// CHECK: remark: Successfully spawned module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Successfully connected to module build daemon at mbd-handshake/mbd.sock [-Rmodule-build-daemon]
// CHECK: remark: Successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// COM: Check that clang invocation can handshake with existing module build daemon
// CHECK-EXIST: remark: Successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
