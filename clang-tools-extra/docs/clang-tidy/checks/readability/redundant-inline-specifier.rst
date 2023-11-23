.. title:: clang-tidy - readability-redundant-inline-specifier

readability-redundant-inline-specifier
======================================

Checks for instances of the ``inline`` keyword in code where it is redundant
and recommends its removal.

Examples:

.. code-block:: c++

   constexpr inline void f() {}

In the example above the keyword ``inline`` is redundant since constexpr
functions are implicitly inlined

.. code-block:: c++
   
   class MyClass {
       inline void myMethod() {}
   };

In the example above the keyword ``inline`` is redundant since member functions
defined entirely inside a class/struct/union definition are implicitly inlined.
