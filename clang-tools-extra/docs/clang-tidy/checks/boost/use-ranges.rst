.. title:: clang-tidy - boost-use-ranges

boost-use-ranges
================

Detects calls to standard library iterator algorithms that could be replaced
with a boost ranges version instead.

Example
-------

.. code-block:: c++

  auto Iter1 = std::find(Items.begin(), Items.end(), 0);
  auto AreSame = std::equal(Items1.cbegin(), Items1.cend(), std::begin(Items2),
                            std::end(Items2));


transforms to:

.. code-block:: c++

  auto Iter1 = boost::range::find(Items, 0);
  auto AreSame = boost::range::equal(Items1, Items2);

Calls to the following std library algorithms are checked:
``includes``,``set_union``,``set_intersection``,``set_difference``,
``set_symmetric_difference``,``unique``,``lower_bound``,``stable_sort``,
``equal_range``,``remove_if``,``sort``,``random_shuffle``,``remove_copy``,
``stable_partition``,``remove_copy_if``,``count``,``copy_backward``,
``reverse_copy``,``adjacent_find``,``remove``,``upper_bound``,``binary_search``,
``replace_copy_if``,``for_each``,``generate``,``count_if``,``min_element``,
``reverse``,``replace_copy``,``fill``,``unique_copy``,``transform``,``copy``,
``replace``,``find``,``replace_if``,``find_if``,``partition``,``max_element``,
``find_end``,``merge``,``partial_sort_copy``,``find_first_of``,``search``,
``lexicographical_compare``,``equal``,``mismatch``,``next_permutation``,
``prev_permutation``,``push_heap``,``pop_heap``,``make_heap``,``sort_heap``,
``copy_if``,``is_permutation``,``is_partitioned``,``find_if_not``,
``partition_copy``,``any_of``,``iota``,``all_of``,``partition_point``,
``is_sorted``,``none_of``,``is_sorted_until``,``reduce``.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: IncludeBoostSystem
   
   If `true` the boost headers are included as system headers with angle
   brackets (`#include <boost.hpp>`), otherwise quotes are used
   (`#include "boost.hpp"`).
