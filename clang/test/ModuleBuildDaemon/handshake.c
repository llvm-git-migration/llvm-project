// COM: Check that clang invocation can spawn and handshake with module build daemon
// COM: Also check that clang invocation can handshake with existing module build daemon

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

// CHECK: remark: Trying to spawn module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Successfully spawned module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Trying to connect to recently spawned module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Succesfully connected to recently spawned module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Trying to send HandshakeMsg to module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Successfully sent HandshakeMsg to module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Waiting to receive module build daemon response [-Rmodule-build-daemon]
// CHECK: remark: Successfully received HandshakeMsg from module build daemon [-Rmodule-build-daemon]
// CHECK: remark: Completed successfull handshake with module build daemon [-Rmodule-build-daemon]

// CHECK-EXIST: remark: Module build daemon already exists [-Rmodule-build-daemon]
// CHECK-EXIST: remark: Trying to send HandshakeMsg to module build daemon [-Rmodule-build-daemon]
// CHECK-EXIST: remark: Successfully sent HandshakeMsg to module build daemon [-Rmodule-build-daemon]
// CHECK-EXIST: remark: Waiting to receive module build daemon response [-Rmodule-build-daemon]
// CHECK-EXIST: remark: Successfully received HandshakeMsg from module build daemon [-Rmodule-build-daemon]
// CHECK-EXIST: remark: Completed successfull handshake with module build daemon [-Rmodule-build-daemon]

// RUN: if pgrep -f "cc1modbuildd mbd-handshake"; then pkill -f "cc1modbuildd mbd-handshake"; fi
