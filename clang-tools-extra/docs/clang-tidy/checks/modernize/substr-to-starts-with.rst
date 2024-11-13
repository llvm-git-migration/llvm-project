modernize-substr-to-starts-with
==============================

Finds calls to ``substr(0, n)`` that can be replaced with ``starts_with()`` (introduced in C++20).
This makes the code's intent clearer and can be more efficient as it avoids creating temporary strings.

For example:

.. code-block:: c++

  str.substr(0, 3) == "foo"     // before
  str.starts_with("foo")        // after

  "bar" == str.substr(0, 3)     // before
  str.starts_with("bar")        // after

  str.substr(0, n) == prefix    // before
  str.starts_with(prefix)       // after

The check handles various ways of expressing zero as the start index:

.. code-block:: c++

  const int zero = 0;
  str.substr(zero, n) == prefix     // converted
  str.substr(x - x, n) == prefix    // converted

The check will only convert cases where:
* The substr call starts at index 0 (or equivalent)
* When comparing with string literals, the length matches exactly
* The comparison is with == or !=

.. code-block:: c++

  auto prefix = str.substr(0, n);    // warns about possible use of starts_with
