// Test without serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ast-dump -ast-dump-filter=Test %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++26 -triple x86_64-unknown-unknown -include-pch %t \
// RUN:             -ast-dump-all -ast-dump-filter=Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

template <template <class> class TT> using N = TT<int>;

namespace qualified {
  namespace foo {
    template <class T> struct A;
  } // namespace foo
  using TestQualified = N<foo::A>;
} // namespace qualified

// CHECK:      Dumping qualified::TestQualified:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TemplateSpecializationType
// CHECK-NEXT:     |-name: 'N' qualified
// CHECK-NEXT:     | `-TypeAliasTemplateDecl {{.+}} N{{$}}
// CHECK-NEXT:     |-TemplateArgument template 'foo::A':'qualified::foo::A' qualified{{$}}
// CHECK-NEXT:     | |-NestedNameSpecifier Namespace 0x{{.+}} 'foo'{{$}}
// CHECK-NEXT:     | `-ClassTemplateDecl {{.+}} A{{$}}

namespace dependent {
  template <class T> struct B {
    using TestDependent = N<T::template X>;
  };
} // namespace dependent

// CHECK:      Dumping dependent::B::TestDependent:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TemplateSpecializationType
// CHECK-NEXT:     |-name: 'N' qualified
// CHECK-NEXT:     | `-TypeAliasTemplateDecl
// CHECK-NEXT:     |-TemplateArgument template 'T::template X':'type-parameter-0-0::template X' dependent{{$}}
// CHECK-NEXT:     | `-NestedNameSpecifier TypeSpec 'T'{{$}}

namespace subst {
  template <class> struct A;

  template <template <class> class TT> struct B {
    template <template <class> class> struct C {};
    using type = C<TT>;
  };
  using TestSubst = B<A>::type;
} // namespace subst

// CHECK:      Dumping subst::TestSubst:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TypedefType
// CHECK-NEXT:     |-TypeAlias
// CHECK-NEXT:     `-ElaboratedType
// CHECK-NEXT:       `-TemplateSpecializationType
// CHECK-NEXT:         |-name: 'C':'subst::B<subst::A>::C' qualified
// CHECK-NEXT:         | `-ClassTemplateDecl {{.+}} C
// CHECK-NEXT:         |-TemplateArgument template 'subst::A' subst index 0
// CHECK-NEXT:         | |-parameter: TemplateTemplateParmDecl {{.+}} depth 0 index 0 TT{{$}}
// CHECK-NEXT:         | |-associated ClassTemplateSpecialization {{.+}} 'B'{{$}}
// CHECK-NEXT:         | `-replacement:
// CHECK-NEXT:         |   `-ClassTemplateDecl {{.+}} A{{$}}

namespace deduced {
  template <class> struct D;

  template <class ET, template <class> class VT>
  struct D<VT<ET>> {
    using E = VT<char>;
    template <class C> using F = VT<C>;
  };

  template <typename, int> class Matrix;

  using TestDeduced1 = D<Matrix<double, 3>>::E;
  using TestDeduced2 = D<Matrix<double, 3>>::F<int>;
} // namespace deduced

// CHECK:      Dumping deduced::TestDeduced1:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TypedefType
// CHECK-NEXT:     |-TypeAlias
// CHECK-NEXT:     `-ElaboratedType
// CHECK-NEXT:       `-TemplateSpecializationType
// CHECK-NEXT:         |-name: 'deduced::Matrix:1<3>' subst index 1
// CHECK-NEXT:         | |-parameter: TemplateTemplateParmDecl {{.+}} depth 0 index 1 VT
// CHECK-NEXT:         | |-associated
// CHECK-NEXT:         | `-replacement: 'deduced::Matrix:1<3>' deduced
// CHECK-NEXT:         |   |-underlying: 'deduced::Matrix'
// CHECK-NEXT:         |   | `-ClassTemplateDecl {{.+}} Matrix
// CHECK-NEXT:         |   `-defaults:  start 1
// CHECK-NEXT:         |     `-TemplateArgument integral '3'
// CHECK-NEXT:         |-TemplateArgument type 'char'
// CHECK-NEXT:         | `-BuiltinType
// CHECK-NEXT:         `-RecordType
// CHECK-NEXT:           `-ClassTemplateSpecialization

// CHECK:      Dumping deduced::TestDeduced2:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TemplateSpecializationType
// CHECK-NEXT:     |-name: 'D<Matrix<double, 3>>::F':'deduced::D<deduced::Matrix<double, 3>>::F
// CHECK-NEXT:     | |-NestedNameSpecifier
// CHECK-NEXT:     | `-TypeAliasTemplateDecl
// CHECK-NEXT:     |-TemplateArgument type 'int'
// CHECK-NEXT:     | `-BuiltinType
// CHECK-NEXT:     `-ElaboratedType
// CHECK-NEXT:       `-TemplateSpecializationType
// CHECK-NEXT:         |-name: 'Matrix:1<3>':'deduced::Matrix:1<3>' deduced
// CHECK-NEXT:         | |-underlying: 'Matrix':'deduced::Matrix' qualified
// CHECK-NEXT:         | | `-ClassTemplateDecl {{.+}} Matrix
// CHECK-NEXT:         | `-defaults:  start 1
// CHECK-NEXT:         |   `-TemplateArgument expr '3'
// CHECK-NEXT:         |-TemplateArgument type 'int'
// CHECK-NEXT:         | `-BuiltinType
// CHECK-NEXT:         `-RecordType
// CHECK-NEXT:           `-ClassTemplateSpecialization
