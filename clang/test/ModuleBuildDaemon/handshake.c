// REQUIRES: !system-windows

//  RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
//  RUN: rm -rf mbd-handshake %t
//  RUN: split-file %s %t

//--- main.c
int main() {return 0;}

// Add '|| true' to ensure RUN command never fails so that daemon shutdown command is always run
// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-new || true
// RUN: %clang -fmodule-build-daemon=mbd-handshake -Rmodule-build-daemon %t/main.c &> %t/output-existing || true
// RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi

// RUN: cat %t/output-new | FileCheck %s
// RUN: cat %t/output-existing | FileCheck %s --check-prefix=CHECK-EXIST

// Check that a clang invocation can spawn and handshake with a module build daemon
// CHECK: remark: Successfully spawned module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Successfully connected to module build daemon at mbd-handshake/mbd.sock [-Rmodule-build-daemon]
// CHECK: remark: Successfully completed handshake with module build daemon [-Rmodule-build-daemon]

// Check that a clang invocation can handshake with an existing module build daemon
// CHECK-EXIST: remark: Successfully completed handshake with module build daemon [-Rmodule-build-daemon]
