===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming Clang |version| release.
     Release notes for previous releases can be found on
     `the Releases Page <https://llvm.org/releases/>`_.

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. For the libc++ release notes,
see `this page <https://libcxx.llvm.org/ReleaseNotes.html>`_. All LLVM releases
may be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

<<<<<<< HEAD
- Fix a bug in reversed argument for templated operators.
  This breaks code in C++20 which was previously accepted in C++17.
  Clang did not properly diagnose such casese in C++20 before this change. Eg:

  .. code-block:: cpp

    struct P {};
    template<class S> bool operator==(const P&, const S&);

    struct A : public P {};
    struct B : public P {};

    // This equality is now ambiguous in C++20.
    bool ambiguous(A a, B b) { return a == b; }

    template<class S> bool operator!=(const P&, const S&);
    // Ok. Found a matching operator!=.
    bool fine(A a, B b) { return a == b; }

  To reduce such widespread breakages, as an extension, Clang accepts this code
  with an existing warning ``-Wambiguous-reversed-operator`` warning.
  Fixes `GH <https://github.com/llvm/llvm-project/issues/53954>`_.

- The CMake variable ``GCC_INSTALL_PREFIX`` (which sets the default
  ``--gcc-toolchain=``) is deprecated and will be removed. Specify
  ``--gcc-install-dir=`` or ``--gcc-triple=`` in a `configuration file
  <https://clang.llvm.org/docs/UsersManual.html#configuration-files>` as a
  replacement.
  (`#77537 <https://github.com/llvm/llvm-project/pull/77537>`_)

C/C++ Language Potentially Breaking Changes
-------------------------------------------

- The default extension name for PCH generation (``-c -xc-header`` and ``-c
  -xc++-header``) is now ``.pch`` instead of ``.gch``.
- ``-include a.h`` probing ``a.h.gch`` will now ignore ``a.h.gch`` if it is not
  a clang pch file or a directory containing any clang pch file.
- Fixed a bug that caused ``__has_cpp_attribute`` and ``__has_c_attribute``
  return incorrect values for some C++-11-style attributes. Below is a complete
  list of behavior changes.

  .. csv-table::
    :header: Test, Old value, New value

    ``__has_cpp_attribute(unused)``,                    201603, 0
    ``__has_cpp_attribute(gnu::unused)``,               201603, 1
    ``__has_c_attribute(unused)``,                      202106, 0
    ``__has_cpp_attribute(clang::fallthrough)``,        201603, 1
    ``__has_cpp_attribute(gnu::fallthrough)``,          201603, 1
    ``__has_c_attribute(gnu::fallthrough)``,            201910, 1
    ``__has_cpp_attribute(warn_unused_result)``,        201907, 0
    ``__has_cpp_attribute(clang::warn_unused_result)``, 201907, 1
    ``__has_cpp_attribute(gnu::warn_unused_result)``,   201907, 1
    ``__has_c_attribute(warn_unused_result)``,          202003, 0
    ``__has_c_attribute(gnu::warn_unused_result)``,     202003, 1

- Fixed a bug in finding matching `operator!=` while adding reversed `operator==` as
  outlined in "The Equality Operator You Are Looking For" (`P2468 <http://wg21.link/p2468r2>`_).
  Fixes (`#68901: <https://github.com/llvm/llvm-project/issues/68901>`_).

- Remove the hardcoded path to the imported modules for C++20 named modules. Now we
  require all the dependent modules to specified from the command line.
  See (`#62707: <https://github.com/llvm/llvm-project/issues/62707>`_).

=======
C/C++ Language Potentially Breaking Changes
-------------------------------------------

>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e
C++ Specific Potentially Breaking Changes
-----------------------------------------

ABI Changes in This Version
---------------------------

AST Dumping Potentially Breaking Changes
----------------------------------------

What's New in Clang |release|?
==============================
Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

C++ Language Changes
--------------------

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++23 Feature Support
^^^^^^^^^^^^^^^^^^^^^

C++2c Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Implemented `P2662R3 Pack Indexing <https://wg21.link/P2662R3>`_.


Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Substitute template parameter pack, when it is not explicitly specified
  in the template parameters, but is deduced from a previous argument.
  (`#78449: <https://github.com/llvm/llvm-project/issues/78449>`_).

C Language Changes
------------------
<<<<<<< HEAD
- ``structs``, ``unions``, and ``arrays`` that are const may now be used as
  constant expressions.  This change is more consistent with the behavior of
  GCC.
- Enums will now be represented in TBAA metadata using their actual underlying
  integer type. Previously they were treated as chars, which meant they could
  alias with all other types.
- Clang now supports the C-only attribute ``counted_by``. When applied to a
  struct's flexible array member, it points to the struct field that holds the
  number of elements in the flexible array member. This information can improve
  the results of the array bound sanitizer and the
  ``__builtin_dynamic_object_size`` builtin.
=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

C23 Feature Support
^^^^^^^^^^^^^^^^^^^

Non-comprehensive list of changes in this release
-------------------------------------------------

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

Attribute Changes in Clang
--------------------------

Improvements to Clang's diagnostics
-----------------------------------
<<<<<<< HEAD
- Clang constexpr evaluator now prints template arguments when displaying
  template-specialization function calls.
- Clang contexpr evaluator now displays notes as well as an error when a constructor
  of a base class is not called in the constructor of its derived class.
- Clang no longer emits ``-Wmissing-variable-declarations`` for variables declared
  with the ``register`` storage class.
- Clang's ``-Wswitch-default`` flag now diagnoses whenever a ``switch`` statement
  does not have a ``default`` label.
- Clang's ``-Wtautological-negation-compare`` flag now diagnoses logical
  tautologies like ``x && !x`` and ``!x || x`` in expressions. This also
  makes ``-Winfinite-recursion`` diagnose more cases.
  (`#56035: <https://github.com/llvm/llvm-project/issues/56035>`_).
- Clang constexpr evaluator now diagnoses compound assignment operators against
  uninitialized variables as a read of uninitialized object.
  (`#51536 <https://github.com/llvm/llvm-project/issues/51536>`_)
- Clang's ``-Wformat-truncation`` now diagnoses ``snprintf`` call that is known to
  result in string truncation.
  (`#64871: <https://github.com/llvm/llvm-project/issues/64871>`_).
  Existing warnings that similarly warn about the overflow in ``sprintf``
  now falls under its own warning group ```-Wformat-overflow`` so that it can
  be disabled separately from ``Wfortify-source``.
  These two new warning groups have subgroups ``-Wformat-truncation-non-kprintf``
  and ``-Wformat-overflow-non-kprintf``, respectively. These subgroups are used when
  the format string contains ``%p`` format specifier.
  Because Linux kernel's codebase has format extensions for ``%p``, kernel developers
  are encouraged to disable these two subgroups by setting ``-Wno-format-truncation-non-kprintf``
  and ``-Wno-format-overflow-non-kprintf`` in order to avoid false positives on
  the kernel codebase.
  Also clang no longer emits false positive warnings about the output length of
  ``%g`` format specifier and about ``%o, %x, %X`` with ``#`` flag.
- Clang now emits ``-Wcast-qual`` for functional-style cast expressions.
- Clang no longer emits irrelevant notes about unsatisfied constraint expressions
  on the left-hand side of ``||`` when the right-hand side constraint is satisfied.
  (`#54678: <https://github.com/llvm/llvm-project/issues/54678>`_).
- Clang now prints its 'note' diagnostic in cyan instead of black, to be more compatible
  with terminals with dark background colors. This is also more consistent with GCC.
- Clang now displays an improved diagnostic and a note when a defaulted special
  member is marked ``constexpr`` in a class with a virtual base class
  (`#64843: <https://github.com/llvm/llvm-project/issues/64843>`_).
- ``-Wfixed-enum-extension`` and ``-Wmicrosoft-fixed-enum`` diagnostics are no longer
  emitted when building as C23, since C23 standardizes support for enums with a
  fixed underlying type.
- When describing the failure of static assertion of `==` expression, clang prints the integer
  representation of the value as well as its character representation when
  the user-provided expression is of character type. If the character is
  non-printable, clang now shows the escpaed character.
  Clang also prints multi-byte characters if the user-provided expression
  is of multi-byte character type.

  *Example Code*:

  .. code-block:: c++

     static_assert("A\n"[1] == U'🌍');

  *BEFORE*:

  .. code-block:: text

    source:1:15: error: static assertion failed due to requirement '"A\n"[1] == U'\U0001f30d''
    1 | static_assert("A\n"[1] == U'🌍');
      |               ^~~~~~~~~~~~~~~~~
    source:1:24: note: expression evaluates to ''
    ' == 127757'
    1 | static_assert("A\n"[1] == U'🌍');
      |               ~~~~~~~~~^~~~~~~~

  *AFTER*:

  .. code-block:: text

    source:1:15: error: static assertion failed due to requirement '"A\n"[1] == U'\U0001f30d''
    1 | static_assert("A\n"[1] == U'🌍');
      |               ^~~~~~~~~~~~~~~~~
    source:1:24: note: expression evaluates to ''\n' (0x0A, 10) == U'🌍' (0x1F30D, 127757)'
    1 | static_assert("A\n"[1] == U'🌍');
      |               ~~~~~~~~~^~~~~~~~
- Clang now always diagnoses when using non-standard layout types in ``offsetof`` .
  (`#64619: <https://github.com/llvm/llvm-project/issues/64619>`_)
- Clang now diagnoses redefined defaulted constructor when redefined
  defaulted constructor with different exception specs.
  (`#69094: <https://github.com/llvm/llvm-project/issues/69094>`_)
- Clang now diagnoses use of variable-length arrays in C++ by default (and
  under ``-Wall`` in GNU++ mode). This is an extension supported by Clang and
  GCC, but is very easy to accidentally use without realizing it's a
  nonportable construct that has different semantics from a constant-sized
  array. (`#62836 <https://github.com/llvm/llvm-project/issues/62836>`_)

- Clang changed the order in which it displays candidate functions on overloading failures.
  Previously, Clang used definition of ordering from the C++ Standard. The order defined in
  the Standard is partial and is not suited for sorting. Instead, Clang now uses a strict
  order that still attempts to push more relevant functions to the top by comparing their
  corresponding conversions. In some cases, this results in better order. E.g., for the
  following code

  .. code-block:: cpp

      struct Foo {
        operator int();
        operator const char*();
      };

      void test() { Foo() - Foo(); }

  Clang now produces a list with two most relevant builtin operators at the top,
  i.e. ``operator-(int, int)`` and ``operator-(const char*, const char*)``.
  Previously ``operator-(const char*, const char*)`` was the first element,
  but ``operator-(int, int)`` was only the 13th element in the output.
  However, new implementation does not take into account some aspects of
  C++ semantics, e.g. which function template is more specialized. This
  can sometimes lead to worse ordering.


- When describing a warning/error in a function-style type conversion Clang underlines only until
  the end of the expression we convert from. Now Clang underlines until the closing parenthesis.

  Before:

  .. code-block:: text

    warning: cast from 'long (*)(const int &)' to 'decltype(fun_ptr)' (aka 'long (*)(int &)') converts to incompatible function type [-Wcast-function-type-strict]
    24 | return decltype(fun_ptr)( f_ptr /*comment*/);
       |        ^~~~~~~~~~~~~~~~~~~~~~~~

  After:

  .. code-block:: text

    warning: cast from 'long (*)(const int &)' to 'decltype(fun_ptr)' (aka 'long (*)(int &)') converts to incompatible function type [-Wcast-function-type-strict]
    24 | return decltype(fun_ptr)( f_ptr /*comment*/);
       |        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``-Wzero-as-null-pointer-constant`` diagnostic is no longer emitted when using ``__null``
  (or, more commonly, ``NULL`` when the platform defines it as ``__null``) to be more consistent
  with GCC.
- Clang will warn on deprecated specializations used in system headers when their instantiation
  is caused by user code.
- Clang will now print ``static_assert`` failure details for arithmetic binary operators.
  Example:

  .. code-block:: cpp

    static_assert(1 << 4 == 15);

  will now print:

  .. code-block:: text

    error: static assertion failed due to requirement '1 << 4 == 15'
       48 | static_assert(1 << 4 == 15);
          |               ^~~~~~~~~~~~
    note: expression evaluates to '16 == 15'
       48 | static_assert(1 << 4 == 15);
          |               ~~~~~~~^~~~~

- Clang now diagnoses definitions of friend function specializations, e.g. ``friend void f<>(int) {}``.
- Clang now diagnoses narrowing conversions involving const references.
  (`#63151: <https://github.com/llvm/llvm-project/issues/63151>`_).
- Clang now diagnoses unexpanded packs within the template argument lists of function template specializations.
- Clang now diagnoses attempts to bind a bitfield to an NTTP of a reference type as erroneous
  converted constant expression and not as a reference to subobject.

=======
- Clang now applies syntax highlighting to the code snippets it
  prints.
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

Improvements to Clang's time-trace
----------------------------------

Bug Fixes in This Version
-------------------------
<<<<<<< HEAD
- Fixed an issue where a class template specialization whose declaration is
  instantiated in one module and whose definition is instantiated in another
  module may end up with members associated with the wrong declaration of the
  class, which can result in miscompiles in some cases.
- Fix crash on use of a variadic overloaded operator.
  (`#42535 <https://github.com/llvm/llvm-project/issues/42535>`_)
- Fix a hang on valid C code passing a function type as an argument to
  ``typeof`` to form a function declaration.
  (`#64713 <https://github.com/llvm/llvm-project/issues/64713>`_)
- Clang now reports missing-field-initializers warning for missing designated
  initializers in C++.
  (`#56628 <https://github.com/llvm/llvm-project/issues/56628>`_)
- Clang now respects ``-fwrapv`` and ``-ftrapv`` for ``__builtin_abs`` and
  ``abs`` builtins.
  (`#45129 <https://github.com/llvm/llvm-project/issues/45129>`_,
  `#45794 <https://github.com/llvm/llvm-project/issues/45794>`_)
- Fixed an issue where accesses to the local variables of a coroutine during
  ``await_suspend`` could be misoptimized, including accesses to the awaiter
  object itself.
  (`#56301 <https://github.com/llvm/llvm-project/issues/56301>`_)
  The current solution may bring performance regressions if the awaiters have
  non-static data members. See
  `#64945 <https://github.com/llvm/llvm-project/issues/64945>`_ for details.
- Clang now prints unnamed members in diagnostic messages instead of giving an
  empty ''. Fixes
  (`#63759 <https://github.com/llvm/llvm-project/issues/63759>`_)
- Fix crash in __builtin_strncmp and related builtins when the size value
  exceeded the maximum value representable by int64_t. Fixes
  (`#64876 <https://github.com/llvm/llvm-project/issues/64876>`_)
- Fixed an assertion if a function has cleanups and fatal erors.
  (`#48974 <https://github.com/llvm/llvm-project/issues/48974>`_)
- Clang now emits an error if it is not possible to deduce array size for a
  variable with incomplete array type.
  (`#37257 <https://github.com/llvm/llvm-project/issues/37257>`_)
- Clang's ``-Wunused-private-field`` no longer warns on fields whose type is
  declared with ``[[maybe_unused]]``.
  (`#61334 <https://github.com/llvm/llvm-project/issues/61334>`_)
- For function multi-versioning using the ``target``, ``target_clones``, or
  ``target_version`` attributes, remove comdat for internal linkage functions.
  (`#65114 <https://github.com/llvm/llvm-project/issues/65114>`_)
- Clang now reports ``-Wformat`` for bool value and char specifier confusion
  in scanf. Fixes
  (`#64987 <https://github.com/llvm/llvm-project/issues/64987>`_)
- Support MSVC predefined macro expressions in constant expressions and in
  local structs.
- Correctly parse non-ascii identifiers that appear immediately after a line splicing
  (`#65156 <https://github.com/llvm/llvm-project/issues/65156>`_)
- Clang no longer considers the loss of ``__unaligned`` qualifier from objects as
  an invalid conversion during method function overload resolution.
- Fix lack of comparison of declRefExpr in ASTStructuralEquivalence
  (`#66047 <https://github.com/llvm/llvm-project/issues/66047>`_)
- Fix parser crash when dealing with ill-formed objective C++ header code. Fixes
  (`#64836 <https://github.com/llvm/llvm-project/issues/64836>`_)
- Fix crash in implicit conversions from initialize list to arrays of unknown
  bound for C++20. Fixes
  (`#62945 <https://github.com/llvm/llvm-project/issues/62945>`_)
- Clang now allows an ``_Atomic`` qualified integer in a switch statement. Fixes
  (`#65557 <https://github.com/llvm/llvm-project/issues/65557>`_)
- Fixes crash when trying to obtain the common sugared type of
  `decltype(instantiation-dependent-expr)`.
  Fixes (`#67603 <https://github.com/llvm/llvm-project/issues/67603>`_)
- Fixes a crash caused by a multidimensional array being captured by a lambda
  (`#67722 <https://github.com/llvm/llvm-project/issues/67722>`_).
- Fixes a crash when instantiating a lambda with requires clause.
  (`#64462 <https://github.com/llvm/llvm-project/issues/64462>`_)
- Fixes a regression where the ``UserDefinedLiteral`` was not properly preserved
  while evaluating consteval functions. (`#63898 <https://github.com/llvm/llvm-project/issues/63898>`_).
- Fix a crash when evaluating value-dependent structured binding
  variables at compile time.
  Fixes (`#67690 <https://github.com/llvm/llvm-project/issues/67690>`_)
- Fixes a ``clang-17`` regression where ``LLVM_UNREACHABLE_OPTIMIZE=OFF``
  cannot be used with ``Release`` mode builds. (`#68237 <https://github.com/llvm/llvm-project/issues/68237>`_).
- Fix crash in evaluating ``constexpr`` value for invalid template function.
  Fixes (`#68542 <https://github.com/llvm/llvm-project/issues/68542>`_)
- Clang will correctly evaluate ``noexcept`` expression for template functions
  of template classes. Fixes
  (`#68543 <https://github.com/llvm/llvm-project/issues/68543>`_,
  `#42496 <https://github.com/llvm/llvm-project/issues/42496>`_,
  `#77071 <https://github.com/llvm/llvm-project/issues/77071>`_,
  `#77411 <https://github.com/llvm/llvm-project/issues/77411>`_)
- Fixed an issue when a shift count larger than ``__INT64_MAX__``, in a right
  shift operation, could result in missing warnings about
  ``shift count >= width of type`` or internal compiler error.
- Fixed an issue with computing the common type for the LHS and RHS of a `?:`
  operator in C. No longer issuing a confusing diagnostic along the lines of
  "incompatible operand types ('foo' and 'foo')" with extensions such as matrix
  types. Fixes (`#69008 <https://github.com/llvm/llvm-project/issues/69008>`_)
- Clang no longer permits using the `_BitInt` types as an underlying type for an
  enumeration as specified in the C23 Standard.
  Fixes (`#69619 <https://github.com/llvm/llvm-project/issues/69619>`_)
- Fixed an issue when a shift count specified by a small constant ``_BitInt()``,
  in a left shift operation, could result in a faulty warnings about
  ``shift count >= width of type``.
- Clang now accepts anonymous members initialized with designated initializers
  inside templates.
  Fixes (`#65143 <https://github.com/llvm/llvm-project/issues/65143>`_)
- Fix crash in formatting the real/imaginary part of a complex lvalue.
  Fixes (`#69218 <https://github.com/llvm/llvm-project/issues/69218>`_)
- No longer use C++ ``thread_local`` semantics in C23 when using
  ``thread_local`` instead of ``_Thread_local``.
  Fixes (`#70068 <https://github.com/llvm/llvm-project/issues/70068>`_) and
  (`#69167 <https://github.com/llvm/llvm-project/issues/69167>`_)
- Fix crash in evaluating invalid lambda expression which forget capture this.
  Fixes (`#67687 <https://github.com/llvm/llvm-project/issues/67687>`_)
- Fix crash from constexpr evaluator evaluating uninitialized arrays as rvalue.
  Fixes (`#67317 <https://github.com/llvm/llvm-project/issues/67317>`_)
- Clang now properly diagnoses use of stand-alone OpenMP directives after a
  label (including ``case`` or ``default`` labels).
- Fix compiler memory leak for enums with underlying type larger than 64 bits.
  Fixes (`#78311 <https://github.com/llvm/llvm-project/pull/78311>`_)

  Before:

  .. code-block:: c++

    label:
    #pragma omp barrier // ok

  After:

  .. code-block:: c++

    label:
    #pragma omp barrier // error: '#pragma omp barrier' cannot be an immediate substatement

- Fixed an issue that a benign assertion might hit when instantiating a pack expansion
  inside a lambda. (`#61460 <https://github.com/llvm/llvm-project/issues/61460>`_)
- Fix crash during instantiation of some class template specializations within class
  templates. Fixes (`#70375 <https://github.com/llvm/llvm-project/issues/70375>`_)
- Fix crash during code generation of C++ coroutine initial suspend when the return
  type of await_resume is not trivially destructible.
  Fixes (`#63803 <https://github.com/llvm/llvm-project/issues/63803>`_)
- ``__is_trivially_relocatable`` no longer returns true for non-object types
  such as references and functions.
  Fixes (`#67498 <https://github.com/llvm/llvm-project/issues/67498>`_)
- Fix crash when the object used as a ``static_assert`` message has ``size`` or ``data`` members
  which are not member functions.
- Support UDLs in ``static_assert`` message.
- Fixed false positive error emitted by clang when performing qualified name
  lookup and the current class instantiation has dependent bases.
  Fixes (`#13826 <https://github.com/llvm/llvm-project/issues/13826>`_)
- Fix a ``clang-17`` regression where a templated friend with constraints is not
  properly applied when its parameters reference an enclosing non-template class.
  Fixes (`#71595 <https://github.com/llvm/llvm-project/issues/71595>`_)
- Fix the name of the ifunc symbol emitted for multiversion functions declared with the
  ``target_clones`` attribute. This addresses a linker error that would otherwise occur
  when these functions are referenced from other TUs.
- Fixes compile error that double colon operator cannot resolve macro with parentheses.
  Fixes (`#64467 <https://github.com/llvm/llvm-project/issues/64467>`_)
- Clang's ``-Wchar-subscripts`` no longer warns on chars whose values are known non-negative constants.
  Fixes (`#18763 <https://github.com/llvm/llvm-project/issues/18763>`_)
- Fix crash due to incorrectly allowing conversion functions in copy elision.
  Fixes (`#39319 <https://github.com/llvm/llvm-project/issues/39319>`_) and
  (`#60182 <https://github.com/llvm/llvm-project/issues/60182>`_) and
  (`#62157 <https://github.com/llvm/llvm-project/issues/62157>`_) and
  (`#64885 <https://github.com/llvm/llvm-project/issues/64885>`_) and
  (`#65568 <https://github.com/llvm/llvm-project/issues/65568>`_)
- Fix an issue where clang doesn't respect detault template arguments that
  are added in a later redeclaration for CTAD.
  Fixes (`#69987 <https://github.com/llvm/llvm-project/issues/69987>`_)
- Fix an issue where CTAD fails for explicit type conversion.
  Fixes (`#64347 <https://github.com/llvm/llvm-project/issues/64347>`_)
- Fix crash when using C++ only tokens like ``::`` in C compiler clang.
  Fixes (`#73559 <https://github.com/llvm/llvm-project/issues/73559>`_)
- Clang now accepts recursive non-dependent calls to functions with deduced
  return type.
  Fixes (`#71015 <https://github.com/llvm/llvm-project/issues/71015>`_)
- Fix assertion failure when initializing union containing struct with
  flexible array member using empty initializer list.
  Fixes (`#77085 <https://github.com/llvm/llvm-project/issues/77085>`_)
- Fix assertion crash due to failed scope restoring caused by too-early VarDecl
  invalidation by invalid initializer Expr.
  Fixes (`#30908 <https://github.com/llvm/llvm-project/issues/30908>`_)
- Clang now emits correct source location for code-coverage regions in `if constexpr`
  and `if consteval` branches.
  Fixes (`#54419 <https://github.com/llvm/llvm-project/issues/54419>`_)
- Fix assertion failure when declaring a template friend function with
  a constrained parameter in a template class that declares a class method
  or lambda at different depth.
  Fixes (`#75426 <https://github.com/llvm/llvm-project/issues/75426>`_)
- Fix an issue where clang cannot find conversion function with template
  parameter when instantiation of template class.
  Fixes (`#77583 <https://github.com/llvm/llvm-project/issues/77583>`_)
- Fix an issue where CTAD fails for function-type/array-type arguments.
  Fixes (`#51710 <https://github.com/llvm/llvm-project/issues/51710>`_)
=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

- Fix crash when using lifetimebound attribute in function with trailing return.
  Fixes (`#73619 <https://github.com/llvm/llvm-project/issues/73619>`_)
- Fix a crash when specializing an out-of-line member function with a default
  parameter where we did an incorrect specialization of the initialization of
  the default parameter.
  Fixes (`#68490 <https://github.com/llvm/llvm-project/issues/68490>`_)
- Fixed a bug where variables referenced by requires-clauses inside
  nested generic lambdas were not properly injected into the constraint scope.
  (`#73418 <https://github.com/llvm/llvm-project/issues/73418>`_)

- Fixed a crash where we lost uninstantiated constraints on placeholder NTTP packs. Fixes:
  (`#63837 <https://github.com/llvm/llvm-project/issues/63837>`_)

- Fixed a regression where clang forgets how to substitute into constraints on template-template
  parameters. Fixes:
  (`#57410 <https://github.com/llvm/llvm-project/issues/57410>`_) and
  (`#76604 <https://github.com/llvm/llvm-project/issues/57410>`_)

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD
- Fixed an import failure of recursive friend class template.
  `Issue 64169 <https://github.com/llvm/llvm-project/issues/64169>`_
- Remove unnecessary RecordLayout computation when importing UnaryOperator. The
  computed RecordLayout is incorrect if fields are not completely imported and
  should not be cached.
  `Issue 64170 <https://github.com/llvm/llvm-project/issues/64170>`_
- Fixed ``hasAnyBase`` not binding nodes in its submatcher.
  (`#65421 <https://github.com/llvm/llvm-project/issues/65421>`_)
- Fixed a bug where RecursiveASTVisitor fails to visit the
  initializer of a bitfield.
  `Issue 64916 <https://github.com/llvm/llvm-project/issues/64916>`_
- Fixed a bug where range-loop-analysis checks for trivial copyability,
  rather than trivial copy-constructibility
  `Issue 47355 <https://github.com/llvm/llvm-project/issues/47355>`_
- Fixed a bug where Template Instantiation failed to handle Lambda Expressions
  with certain types of Attributes.
  (`#76521 <https://github.com/llvm/llvm-project/issues/76521>`_)
=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<<<<<<< HEAD
- Fixed a crash when parsing top-level ObjC blocks that aren't properly
  terminated. Clang should now also recover better when an @end is missing
  between blocks.
  `Issue 64065 <https://github.com/llvm/llvm-project/issues/64065>`_
- Fixed a crash when check array access on zero-length element.
  `Issue 64564 <https://github.com/llvm/llvm-project/issues/64564>`_
- Fixed a crash when an ObjC ivar has an invalid type. See
  (`#68001 <https://github.com/llvm/llvm-project/pull/68001>`_)
- Fixed a crash in C when redefined struct is another nested redefinition.
  `Issue 41302 <https://github.com/llvm/llvm-project/issues/41302>`_
- Fixed a crash when ``-ast-dump=json`` was used for code using class
  template deduction guides.
- Fixed a crash when a lambda marked as ``static`` referenced a captured
  variable in an expression.
  `Issue 74608 <https://github.com/llvm/llvm-project/issues/74608>`_
- Fixed a crash with modules and a ``constexpr`` destructor.
  `Issue 68702 <https://github.com/llvm/llvm-project/issues/68702>`_

=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

<<<<<<< HEAD
- Added option ``-m[no-]evex512`` to disable ZMM and 64-bit mask instructions
  for AVX512 features.
- Support ISA of ``USER_MSR``.
  * Support intrinsic of ``_urdmsr``.
  * Support intrinsic of ``_uwrmsr``.
- Support ISA of ``AVX10.1``.
- ``-march=pantherlake`` and ``-march=clearwaterforest`` are now supported.
- Added ABI handling for ``__float128`` to match with GCC.
- Emit warnings for options to enable knl/knm specific ISAs: AVX512PF, AVX512ER
  and PREFETCHWT1. From next version (LLVM 19), these ISAs' intrinsic supports
  will be deprecated:
  * intrinsic series of *_exp2a23_*
  * intrinsic series of *_rsqrt28_*
  * intrinsic series of *_rcp28_*
  * intrinsic series of *_prefetch_i[3|6][2|4]gather_*
  * intrinsic series of *_prefetch_i[3|6][2|4]scatter_*

=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e
Arm and AArch64 Support
^^^^^^^^^^^^^^^^^^^^^^^

Android Support
^^^^^^^^^^^^^^^

Windows Support
^^^^^^^^^^^^^^^

LoongArch Support
^^^^^^^^^^^^^^^^^

RISC-V Support
^^^^^^^^^^^^^^
<<<<<<< HEAD
- Unaligned memory accesses can be toggled by ``-m[no-]unaligned-access`` or the
  aliases ``-m[no-]strict-align``.
- CodeGen of RV32E/RV64E was supported experimentally.
- CodeGen of ilp32e/lp64e was supported experimentally.
=======
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

- ``__attribute__((rvv_vector_bits(N)))`` is now supported for RVV vbool*_t types.

CUDA/HIP Language Changes
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA Support
^^^^^^^^^^^^

AIX Support
^^^^^^^^^^^

WebAssembly Support
^^^^^^^^^^^^^^^^^^^

AVR Support
^^^^^^^^^^^

DWARF Support in Clang
----------------------

Floating Point Support in Clang
-------------------------------

AST Matchers
------------
<<<<<<< HEAD
- Add ``convertVectorExpr``.
- Add ``dependentSizedExtVectorType``.
- Add ``macroQualifiedType``.
- Add ``CXXFoldExpr`` related matchers: ``cxxFoldExpr``, ``callee``,
  ``hasInit``, ``hasPattern``, ``isRightFold``, ``isLeftFold``,
  ``isUnaryFold``, ``isBinaryFold``, ``hasOperator``, ``hasLHS``, ``hasRHS``, ``hasEitherOperand``.

clang-format
------------
- Add ``AllowBreakBeforeNoexceptSpecifier`` option.
- Add ``AllowShortCompoundRequirementOnASingleLine`` option.
- Change ``BreakAfterAttributes`` from ``Never`` to ``Leave`` in LLVM style.
- Add ``BreakAdjacentStringLiterals`` option.
- Add ``ObjCPropertyAttributeOrder`` which can be used to sort ObjC property
  attributes (like ``nonatomic, strong, nullable``).
- Add ``PenaltyBreakScopeResolution`` option.
- Add ``.clang-format-ignore`` files.
- Add ``AlignFunctionPointers`` sub-option for ``AlignConsecutiveDeclarations``.
=======

clang-format
------------
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

libclang
--------

Static Analyzer
---------------

New features
^^^^^^^^^^^^

Crash and bug fixes
^^^^^^^^^^^^^^^^^^^

Improvements
^^^^^^^^^^^^

<<<<<<< HEAD
- Improved the ``unix.StdCLibraryFunctions`` checker by modeling more
  functions like ``send``, ``recv``, ``readlink``, ``fflush``, ``mkdtemp``,
  ``getcwd`` and ``errno`` behavior.
  (`52ac71f92d38 <https://github.com/llvm/llvm-project/commit/52ac71f92d38f75df5cb88e9c090ac5fd5a71548>`_,
  `#77040 <https://github.com/llvm/llvm-project/pull/77040>`_,
  `#76671 <https://github.com/llvm/llvm-project/pull/76671>`_,
  `#71373 <https://github.com/llvm/llvm-project/pull/71373>`_,
  `#76557 <https://github.com/llvm/llvm-project/pull/76557>`_,
  `#71392 <https://github.com/llvm/llvm-project/pull/71392>`_)

- Fixed a false negative for when accessing a nonnull property (ObjC).
  (`1dceba3a3684 <https://github.com/llvm/llvm-project/commit/1dceba3a3684d12394731e09a6cf3efcebf07a3a>`_)

- ``security.insecureAPI.DeprecatedOrUnsafeBufferHandling`` now considers
  ``fprintf`` calls unsafe.
  `Documentation <https://clang.llvm.org/docs/analyzer/checkers.html#security-insecureapi-deprecatedorunsafebufferhandling-c>`__.

- Improved the diagnostics of the ``optin.core.EnumCastOutOfRange`` checker.
  It will display the name and the declaration of the enumeration along with
  the concrete value being cast to the enum.
  (`#74503 <https://github.com/llvm/llvm-project/pull/74503>`_)

- Improved the ``alpha.security.ArrayBoundV2`` checker for detecting buffer
  accesses prior the buffer; and also reworked the diagnostic messages.
  (`3e014038b373 <https://github.com/llvm/llvm-project/commit/3e014038b373e5a4a96d89d46cea17e4d2456a04>`_,
  `#70056 <https://github.com/llvm/llvm-project/pull/70056>`_,
  `#72107 <https://github.com/llvm/llvm-project/pull/72107>`_)

- Improved the ``alpha.unix.cstring.OutOfBounds`` checking both ends of the
  buffers in more cases.
  (`c3a87ddad62a <https://github.com/llvm/llvm-project/commit/c3a87ddad62a6cc01acaccc76592bc6730c8ac3c>`_,
  `0954dc3fb921 <https://github.com/llvm/llvm-project/commit/0954dc3fb9214b994623f5306473de075f8e3593>`_)

- Improved the ``alpha.unix.Stream`` checker by modeling more functions
  ``fputs``, ``fputc``, ``fgets``, ``fgetc``, ``fdopen``, ``ungetc``, ``fflush``
  and no not recognize alternative ``fopen`` and ``tmpfile`` implementations.
  (`#76776 <https://github.com/llvm/llvm-project/pull/76776>`_,
  `#74296 <https://github.com/llvm/llvm-project/pull/74296>`_,
  `#73335 <https://github.com/llvm/llvm-project/pull/73335>`_,
  `#72627 <https://github.com/llvm/llvm-project/pull/72627>`_,
  `#71518 <https://github.com/llvm/llvm-project/pull/71518>`_,
  `#72016 <https://github.com/llvm/llvm-project/pull/72016>`_,
  `#70540 <https://github.com/llvm/llvm-project/pull/70540>`_,
  `#73638 <https://github.com/llvm/llvm-project/pull/73638>`_,
  `#77331 <https://github.com/llvm/llvm-project/pull/77331>`_)

- The ``alpha.security.taint.TaintPropagation`` checker no longer propagates
  taint on ``strlen`` and ``strnlen`` calls, unless these are marked
  explicitly propagators in the user-provided taint configuration file.
  This removal empirically reduces the number of false positive reports.
  Read the PR for the details.
  (`#66086 <https://github.com/llvm/llvm-project/pull/66086>`_)

- Other taint-related improvements.
  (`#66358 <https://github.com/llvm/llvm-project/pull/66358>`_,
  `#66074 <https://github.com/llvm/llvm-project/pull/66074>`_,
  `#66358 <https://github.com/llvm/llvm-project/pull/66358>`_)

- Checkers can query constraint bounds to improve diagnostic messages.
  (`#74141 <https://github.com/llvm/llvm-project/pull/74141>`_)
=======
- Support importing C++20 modules in clang-repl.
>>>>>>> faf555f93f3628b7b2b64162c02dd1474540532e

Moved checkers
^^^^^^^^^^^^^^

.. _release-notes-sanitizers:

Sanitizers
----------

Python Binding Changes
----------------------

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
