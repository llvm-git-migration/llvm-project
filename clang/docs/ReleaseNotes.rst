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

C/C++ Language Potentially Breaking Changes
-------------------------------------------

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

Resolutions to C++ Defect Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C Language Changes
------------------

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

Improvements to Clang's time-trace
----------------------------------

Bug Fixes in This Version
-------------------------
<<<<<<< HEAD
=======
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
- Fix crashes when using the binding decl from an invalid structured binding.
  Fixes (`#67495 <https://github.com/llvm/llvm-project/issues/67495>`_) and
  (`#72198 <https://github.com/llvm/llvm-project/issues/72198>`_)
- Fix assertion failure when call noreturn-attribute function with musttail
  attribute.
  Fixes (`#76631 <https://github.com/llvm/llvm-project/issues/76631>`_)
  - The MS ``__noop`` builtin without an argument list is now accepted
  in the placement-args of new-expressions, matching MSVC's behaviour.
- Fix an issue that caused MS ``__decspec(property)`` accesses as well as
  Objective-C++ property accesses to not be converted to a function call
  to the getter in the placement-args of new-expressions.
  Fixes (`#65053 <https://github.com/llvm/llvm-project/issues/65053>`_)
- Fix an issue with missing symbol definitions when the first coroutine
  statement appears in a discarded ``if constexpr`` branch.
  Fixes (`#78290 <https://github.com/llvm/llvm-project/issues/78290>`_)
- Fix crash when using lifetimebound attribute in function with trailing return.
  Fixes (`#73619 <https://github.com/llvm/llvm-project/issues/73619>`_)
>>>>>>> 0e74e6cc4d33 ([Clang][Sema] fix crash of attribute transform)

Bug Fixes to Compiler Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to Attribute Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to C++ Support
^^^^^^^^^^^^^^^^^^^^^^^^

Bug Fixes to AST Handling
^^^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^

Miscellaneous Clang Crashes Fixed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenACC Specific Changes
------------------------

Target Specific Changes
-----------------------

AMDGPU Support
^^^^^^^^^^^^^^

X86 Support
^^^^^^^^^^^

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

clang-format
------------

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

- Support importing C++20 modules in clang-repl.

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
