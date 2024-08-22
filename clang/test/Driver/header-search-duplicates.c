// Test that the gcc-like driver, when not in Microsoft compatibility mode,
// emulates the gcc behavior of eliding a user header search path when the
// same path is present as a system header search path.
// See microsoft-header-search-duplicates.c for Microsoft compatible behavior.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test the clang driver with a target that does not implicitly enable the
// -fms-compatibility option. The -nostdinc option is used to suppress default
// search paths to ease testing.
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu \
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

#--- test.c
#include <a.h>
#include <b.h>
#include <c.h>

// The expected behavior is that user search paths are ordered before system
// search paths, that user search paths that duplicate a (later) system search
// path are ignored, and that search paths that duplicate an earlier search
// path of the same user/system kind are ignored.
// CHECK:      ignoring duplicate directory "[[PWD]]/include/w"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/x"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/z"
// CHECK:      #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/include/w
// CHECK-NEXT: [[PWD]]/include/z
// CHECK-NEXT: [[PWD]]/include/y
// CHECK-NEXT: [[PWD]]/include/x
// CHECK-NEXT: End of search list.

#--- include/w/b.h
#define INCLUDE_W_B_H
#include_next <b.h>

#--- include/w/c.h
#define INCLUDE_W_C_H
#include_next <c.h>

#--- include/x/a.h
#if !defined(INCLUDE_Y_A_H)
#error 'include/y/a.h' should have been included before 'include/x/a.h'!
#endif

#--- include/x/b.h
#if !defined(INCLUDE_W_B_H)
#error 'include/w/b.h' should have been included before 'include/x/b.h'!
#endif

#--- include/x/c.h
#if !defined(INCLUDE_Z_C_H)
#error 'include/z/c.h' should have been included before 'include/x/c.h'!
#endif

#--- include/y/a.h
#define INCLUDE_Y_A_H
#include_next <a.h>

#--- include/z/c.h
#define INCLUDE_Z_C_H
#include_next <c.h>
