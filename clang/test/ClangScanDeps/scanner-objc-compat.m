// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -c %t/main.m -o %t/main.o -fmodules -I %t > %t/deps.db
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Verify that the scanner does not treat ObjC method decls with arguments named
// module or import as module/import decls.

// CHECK: "module-name": "A"

//--- A.h

@interface SomeObjcClass
  - (void)func:(int)otherData
          module:(int)module
          import:(int)import;
@end

//--- module.modulemap

module A {
  header "A.h"
}

//--- main.m

#include "A.h"
