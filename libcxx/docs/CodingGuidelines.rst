.. _CodingGuidelines:

========================
libc++ Coding Guidelines
========================

Use ``__ugly_names`` for implementation details
===============================================

Libc++ uses ``__ugly_names`` for implementation details. These names are reserved for implementations, so users may not
use them in their own applications. When using a name like ``T``, a user may have defined a macro that changes the
meaning of ``T``. By using ``__ugly_names`` we avoid that problem. Other standard libraries and compilers use these
names too.

This is partially enforced by the clangh-tidy check ``readability-identifier-naming`` and
``libcxx/test/libcxx/system_reserved_names.gen.py``.

Don't use argument-dependent lookup unless required by the standard
===================================================================

Unqualified function calls are susceptible to
`argument-dependent lookup (ADL) <https://en.cppreference.com/w/cpp/language/adl>`_. This means calling
``move(UserType)`` might not call ``std::move``. Therefore, function calls must use qualified names to avoid ADL. Some
functions in the standard library `require ADL usage <http://eel.is/c++draft/contents#3>`_. Names of classes, variables,
concepts, and type aliases are not subject to ADL. They don't need to be qualified.

Function overloading also applies to operators. Using ``&user_object`` may call a user-defined ``operator&``. Use
``std::addressof`` instead. Similarly, to avoid invoking a user-defined ``operator,``, make sure to cast the result to
``void`` when using the ``,`` or avoid it in the first place. For example:

.. code-block:: cpp

    for (; __first1 != __last1; ++__first1, (void)++__first2) {
      ...
    }

This is mostly enforced by the clang-tidy checks ``libcpp-robust-against-adl`` and ``libcpp-qualify-declval``.

Avoid including public headers
==============================

libc++ uses implementation-detail headers for most code. These are either top-level headers starting with two
underscores (e.g. ``<__locale>``) or are in a directory that starts with two underscores
(e.g. ``<__type_traits/decay.h>``). These detail headers are significantly smaller than their public counterparts.
This reduces the amount of code that is included in a single public header, reducing compile times in turn.

Add ``_LIBCPP_HIDE_FROM_ABI`` unless you know better
====================================================

``_LIBCPP_HIDE_FROM_ABI`` should be on every function in the library unless there is a reason not to do so. The main
resason to not add ``_LIBCPP_HIDE_FROM_ABI`` is if a function is exported libc++ dylib. In that case a function should
be marked ``_LIBCPP_EXPORTED_FROM_ABI``. Virtual functions should be marked ``_LIBCPP_HIDE_FROM_ABI_VIRTUAL`` instead.

This is mostly enforced by the clang-tidy checks ``libcpp-hide-from-abi`` and ``libcpp-avoid-abi-tag-on-virtual``.

Always define macros
====================

Macros should usually be defined in all configurations. This makes it significantly easier to catch missing includes,
since Clang and GCC will warn when using and undefined marco inside an ``#if`` statement when using ``-Wundef``. Some
macros in libc++ don't use this style yet, so this only applies when introducing a new macro.

This is partially enforced by the clang-tidy check ``libcpp-internal-ftms``.

Use ``_LIBCPP_STD_VER``
=======================

libc++ defines the macro ``_LIBCPP_STD_VER`` for the different libc++ dialects. This should be used instead of
``__cplusplus``. The form ``_LIBCPP_STD_VER >= <version>`` is preferred over ``_LIBCPP_STD_VER > <previous-version>``.

This is mostly enforced by the clang-tidy check ``libcpp-cpp-version-check``.

Use \_\_ugly\_\_ spellings of vendor attributes
===============================================

Vendor attributes should always be \_\_uglified\_\_ to avoid naming clashes with user-defined macros. For gnu-style
attributes this takes the form ``__attribute__((__attribute__))``. C++11-style attributes look like
``[[_Clang::__attribute__]]`` or ``[[__gnu__::__attribute__]]`` for Clang or GCC attributes respectively. Clang and GCC
also support standard attributes in earlier language dialects than they were introduced. These should be spelled as
``[[__attribute__]]`` in these cases. MSVC currently doesn't provide alternative spellings for their attributes, so
these should be avoided if at all possible.

This is enforced by the clang-tidy check ``libcpp-uglify-attributes``.

Use C++11 extensions in C++03 code if they simplify the code
============================================================

libc++ only supports Clang in C++98/03 mode. Clang provides many C++11 features in C++03, making it possible to write a
lot of code in a simpler way than if we were restricted to C++03 features. Some use of extensions is even mandatory,
since libc++ supports move semantics in C++03.

Use ``using`` aliases instead of ``typedef``
============================================

``using`` aliases are generally easier to read and support templates. Some code in libc++ uses ``typedef`` for
historical reasons.

Write SFINAE with ``requires`` clauses in C++20-only code
=========================================================

``requires`` clauses can be significantly easier to read than ``enable_if`` and friends in some cases, since concepts
subsume other concepts. This means that overloads based on traits can be written without negating more general cases.
They also show intent better.

For example
.. code-block:: cpp

  template <class _Iter>
    requires forward_iterator<_Iter>
  void func(_Iter);

  template <class _Iter>
    requires bidirectional_iterator<_Iter>
  void func(_Iter);

is perfectly fine code, but ``enable_if`` would need ``!forward_iterator<_Iter> && bidirectional_iterator<_Iter>`` for
the second overload.

Write ``enable_if``s as ``enable_if_t<conditon, int> = 0``
==========================================================

The form ``enable_if_t<condition, int> = 0`` is the only form that works in every language mode and for overload sets
using the same template arguments otherwise. If the code must work in C++11 or C++03, the libc++-internal alias
``__enable_if_t`` can be used instead.

Prefer alias templates over class templates
===========================================

Alias templates are much more light weight than class templates, since they don't require new instantiations for
different types. They do force more eager evaluation though, which can be a problem in some cases.

Use ``unique_ptr`` when allocating memory
=========================================

The standard library often needs to allocate memory and then construct a user type in it. If the users constructor
throws, the library needs to deallocate that memory. The idiomatic way to achieve this is with ``unique_ptr``.

Apply ``[[nodiscard]]`` liberally
=================================

Libc++ adds ``[[nodiscard]]`` to functions in a lot of places. The standards committee has decided to not have a
recommended practice where to put them, so libc++ has its own guidelines on when to apply ``[[nodiscard]]``.

When should ``[[nodiscard]]`` be added?
---------------------------------------

``[[nodiscard]]`` should be applied to functions

- where discarding the return value is most likely a correctness issue. For example a locking constructor in
  ``unique_lock``.

- where discarding the return value likely points to the user wanting to do something different. For example
  ``vector::empty()``, which probably should have been ``vector::clear()``.

  This can help spotting bugs easily which otherwise may take a very long time to find.

- which return a constant. For example ``numeric_limits::min()``.
- which only observe a value. For example ``string::size()``.

  Code that discards values from these kinds of functions is dead code. It can either be removed, or the programmer
  meant to do something different.

- where discarding the value is most likely a misuse of the function. For example ``find``.

  This protects programmers from assuming too much about how the internals of a function work, making code more robust
  in the presence of future optimizations.

What should be done when adding ``[[nodiscard]]`` to a function?
----------------------------------------------------------------

Applications of ``[[nodiscard]]`` are code like any other code, so we aim to test them. This can be done with a
``.verify.cpp`` test. Many examples are available. Just look for tests with the suffix ``.nodiscard.verify.cpp``.
