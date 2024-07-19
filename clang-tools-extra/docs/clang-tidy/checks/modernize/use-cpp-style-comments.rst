.. title:: clang-tidy - modernize-use-cpp-style-comments

modernize-use-cpp-style-comments
================================

Finds C-style comments and suggests to use C++ style comments `//`.


.. code-block:: c++

  memcpy(a, b, sizeof(int) * 5); /* use std::copy_n instead of memcpy */
  // warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]