.. title:: clang-tidy - readability-redundant-inline-specifier

readability-redundant-inline-specifier
=================

Checks for instances of the `inline` keyword in code where it is redundant
and recommends its removal.

Examples:

.. code-block:: c++

   constexpr inline void f() {}

In the example abvove the keyword `inline` is redundant since constexpr
functions are implicitly inlined

.. code-block:: c++
   class MyClass {
       inline void myMethod() {}
   };

In the example above the keyword `inline` is redundant since member functions
defined entirely inside a class/struct/union definition are implicitly inlined.

The token `inline` is considered redundant in the following cases:

- When it is used in a function definition that is constexpr.
- When it is used in a member function definition that is defined entirely
  inside a class/struct/union definition.
- When it is used on a deleted function. 
- When it is used on a template declaration.
- When it is used on a member variable that is constexpr and static.

