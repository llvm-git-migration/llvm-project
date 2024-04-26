// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: cp a1.h a.h
// RUN: %clang -fmodules -fmodules-validate-input-files-content -Xclang -fno-pch-timestamp -fimplicit-modules -fmodule-map-file=module.modulemap -fsyntax-only  -fmodules-cache-path=%t test1.cpp
// RUN: cp a2.h a.h
// RUN: %clang -fmodules -fmodules-validate-input-files-content -Xclang -fno-pch-timestamp -fimplicit-modules -fmodule-map-file=module.modulemap -fsyntax-only  -fmodules-cache-path=%t test2.cpp

//--- a1.h
#define FOO

//--- a2.h
#define BAR

//--- module.modulemap
module a {
  header "a.h"
}

//--- test1.cpp
#include "a.h"

#ifndef FOO
#error foo
#endif

//--- test2.cpp
#include "a.h"

#ifndef BAR
#error bar
#endif
