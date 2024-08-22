// Test that the cl-like driver and the gcc-like driver, when in Microsoft
// compatibility mode, retain user header search paths that are duplicated in
// system header search paths.
// See header-search-duplicates.c for gcc compatible behavior.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test the clang driver with a Windows target that implicitly enables the
// -fms-compatibility option. The -nostdinc option is used to suppress default
// search paths to ease testing.
// RUN: %clang \
// RUN:     -target x86_64-pc-windows \
// RUN:     -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/include/w \
// RUN:     -isystem %t/include/z \
// RUN:     -I%t/include/x \
// RUN:     -isystem %t/include/y \
// RUN:     -isystem %t/include/x \
// RUN:     -I%t/include/w \
// RUN:     -isystem %t/include/y \
// RUN:     -isystem %t/include/z \
// RUN:     %t/test.c 2>&1 | FileCheck -DPWD=%t %t/test.c

// Test the clang-cl driver with a Windows target that implicitly enables the
// -fms-compatibility option. The /X option is used instead of -nostdinc
// because the latter option suppresses all system include paths including
// those specified by -imsvc. The -nobuiltininc option is uesd to suppress
// the Clang resource directory. The -nostdlibinc option is used to suppress
// search paths for the Windows SDK, CRT, MFC, ATL, etc...
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows \
// RUN:     -v -fsyntax-only \
// RUN:     /X \
// RUN:     -nobuiltininc \
// RUN:     -nostdlibinc \
// RUN:     -I%t/include/w \
// RUN:     -imsvc %t/include/z \
// RUN:     -I%t/include/x \
// RUN:     -imsvc %t/include/y \
// RUN:     -imsvc %t/include/x \
// RUN:     -I%t/include/w \
// RUN:     -imsvc %t/include/y \
// RUN:     -imsvc %t/include/z \
// RUN:     %t/test.c 2>&1 | FileCheck -DPWD=%t %t/test.c

#--- test.c
#include <a.h>
#include <b.h>
#include <c.h>

// The expected behavior is that user search paths are ordered before system
// search paths and that search paths that duplicate an earlier search path
// (regardless of user/system concerns) are ignored.
// CHECK:      ignoring duplicate directory "[[PWD]]/include/w"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/x"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/z"
// CHECK-NOT:   as it is a non-system directory that duplicates a system directory
// CHECK:      #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/include/w
// CHECK-NEXT: [[PWD]]/include/x
// CHECK-NEXT: [[PWD]]/include/z
// CHECK-NEXT: [[PWD]]/include/y
// CHECK-NEXT: End of search list.

#--- include/w/b.h
#define INCLUDE_W_B_H
#include_next <b.h>

#--- include/w/c.h
#define INCLUDE_W_C_H
#include_next <c.h>

#--- include/x/a.h
#define INCLUDE_X_A_H
#include_next <a.h>

#--- include/x/b.h
#if !defined(INCLUDE_W_B_H)
#error 'include/w/b.h' should have been included before 'include/x/b.h'!
#endif

#--- include/x/c.h
#define INCLUDE_X_C_H

#--- include/y/a.h
#if !defined(INCLUDE_X_A_H)
#error 'include/x/a.h' should have been included before 'include/y/a.h'!
#endif

#--- include/z/c.h
#include_next <c.h>
#if !defined(INCLUDE_X_C_H)
#error 'include/x/c.h' should have been included before 'include/z/c.h'!
#endif
