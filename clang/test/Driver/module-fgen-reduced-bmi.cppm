// It is annoying to handle different slash direction
// in Windows and Linux. So we disable the test on Windows
// here.
// REQUIRES: !system-windows
// On AIX, the default output for `-c` may be `.s` instead of `.o`,
// which makes the test fail. So disable the test on AIX.
// REQUIRES: !system-aix
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 %t/Hello.cppm -fmodule-output=%t/Hello.pcm \
// RUN:     -fgen-reduced-bmi -c -o %t/Hello.o -### 2>&1 | FileCheck %t/Hello.cppm
//
// RUN: %clang -std=c++20 %t/Hello.cppm \
// RUN:     -fgen-reduced-bmi -c -o %t/Hello.o -### 2>&1 | \
// RUN:         FileCheck %t/Hello.cppm --check-prefix=CHECK-UNSPECIFIED
//
// RUN: %clang -std=c++20 %t/Hello.cppm \
// RUN:     -fgen-reduced-bmi -c -### 2>&1 | \
// RUN:         FileCheck %t/Hello.cppm --check-prefix=CHECK-NO-O
//
// RUN: %clang -std=c++20 %t/Hello.cppm \
// RUN:     -fgen-reduced-bmi -c -o %t/AnotherName.o -### 2>&1 | \
// RUN:         FileCheck %t/Hello.cppm --check-prefix=CHECK-ANOTHER-NAME
//
// RUN: %clang -std=c++20 %t/Hello.cppm --precompile -fgen-reduced-bmi \
// RUN:     -o %t/Hello.full.pcm -### 2>&1 | FileCheck %t/Hello.cppm \
// RUN:     --check-prefix=CHECK-EMIT-MODULE-INTERFACE
//
// RUN: %clang -std=c++20 %t/Hello.cc -fgen-reduced-bmi -Wall -Werror \
// RUN:     -c -o %t/Hello.o -### 2>&1 | FileCheck %t/Hello.cc

//--- Hello.cppm
export module Hello;

// Test that we won't generate the emit-module-interface as 2 phase compilation model.
// CHECK-NOT: -emit-module-interface
// CHECK: "-fgen-reduced-bmi"

// CHECK-UNSPECIFIED: -fmodule-output={{.*}}/Hello.pcm

// CHECK-NO-O: -fmodule-output={{.*}}/Hello.pcm
// CHECK-ANOTHER-NAME: -fmodule-output={{.*}}/AnotherName.pcm

// With `-emit-module-interface` specified, we should still see the `-emit-module-interface`
// flag.
// CHECK-EMIT-MODULE-INTERFACE: -emit-module-interface

//--- Hello.cc

// CHECK-NOT: "-fgen-reduced-bmi"
