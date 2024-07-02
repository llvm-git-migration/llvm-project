.. title:: clang-tidy - modernize-use-ranges

modernize-use-ranges
====================

Detects calls to standard library iterator algorithms that could be replaced
with a ranges version instead

Example
-------

.. code-block:: c++

  auto Iter1 = std::find(Items.begin(), Items.end(), 0);
  auto AreSame = std::equal(std::execution::par, Items1.cbegin(), Items1.cend(),
                            std::begin(Items2), std::end(Items2));


transforms to:

.. code-block:: c++

  auto Iter1 = std::ranges::find(Items, 0);
  auto AreSame = std::equal(std::execution::par, Items1, Items2);
