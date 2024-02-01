// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -I %t %t/A.cppm -emit-module-interface -o %t/A.pcm -verify
// RUN: %clang_cc1 -std=c++20 -I %t %t/B.cpp -fmodule-file=A=%t/A.pcm -fsyntax-only -verify -ast-dump-all -ast-dump-filter baz | FileCheck %s

//--- foo.h
namespace baz {
  using foo = char;
  using baz::foo;
}

//--- A.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module A;

//--- B.cpp
// expected-no-diagnostics
#include "foo.h"
import A;
// Since modules are loaded lazily, force loading by performing a lookup.
using xxx = baz::foo;

// CHECK-LABEL: Dumping baz:
// CHECK-NEXT: NamespaceDecl 0x[[BAZ_REDECL_ADDR:[^ ]*]] prev 0x[[BAZ_ADDR:[^ ]*]] <{{.*}}> line:{{.*}} imported in A.<global> hidden <undeserialized declarations> baz
// CHECK-NEXT: |-original Namespace 0x[[BAZ_ADDR]] 'baz'
// CHECK-NEXT: |-TypeAliasDecl 0x[[ALIAS_REDECL_ADDR:[^ ]*]] prev 0x[[ALIAS_ADDR:[^ ]*]] <{{.*}}> col:{{.*}} imported in A.<global> hidden foo 'char'
// CHECK-NEXT: | `-BuiltinType 0x{{[^ ]*}} 'char'
// CHECK-NEXT: |-UsingDecl 0x{{[^ ]*}} first 0x[[USING_ADDR:[^ ]*]] <{{.*}}> col:{{.*}} imported in A.<global> hidden baz::foo
// CHECK-NEXT: | `-NestedNameSpecifier Namespace 0x[[BAZ_REDECL_ADDR]] 'baz'
// CHECK-NEXT: `-UsingShadowDecl 0x{{[^ ]*}} prev 0x[[SHADOW_ADDR:[^ ]*]] <{{.*}}> col:{{.*}} imported in A.<global> hidden implicit TypeAlias 0x[[ALIAS_REDECL_ADDR]] 'foo'

// CHECK-LABEL: Dumping baz:
// CHECK-NEXT: NamespaceDecl 0x[[BAZ_ADDR]] <{{.*}}> line:{{.*}} baz
// CHECK-NEXT: |-TypeAliasDecl 0x[[ALIAS_ADDR]] <{{.*}}> col:{{.*}} referenced foo 'char'
// CHECK-NEXT: | `-BuiltinType 0x{{[^ ]*}} 'char'
// CHECK-NEXT: |-UsingDecl 0x[[USING_ADDR]] <{{.*}}> col:{{.*}} baz::foo
// CHECK-NEXT: | `-NestedNameSpecifier Namespace 0x[[BAZ_ADDR]] 'baz'
// CHECK-NEXT:  `-UsingShadowDecl 0x[[SHADOW_ADDR]] <{{.*}}> col:{{.*}} implicit TypeAlias 0x[[ALIAS_ADDR]] 'foo'
