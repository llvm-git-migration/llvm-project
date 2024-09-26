.. title:: clang-tidy - bugprone-nondeterministic-pointer-usage

nondeterministic-pointer-usage
==============================

Finds nondeterministic usages of pointers in unordered containers.

One canonical example is iteration across a container of pointers.

.. code-block:: c++

  {
    for (auto i : UnorderedPtrSet)
      f(i);
  }

Another such example is sorting a container of pointers.

.. code-block:: c++

  {
    std::sort(VectorOfPtr.begin(), VectorOfPtr.end());
  }

Iteration of a containers of pointers may present the order of different
pointers differently across different runs of a program. In some cases this
may be acceptable behavior, in others this may be unexpected behavior. This
check is advisory for this reason.

This check only detects range-based for loops over unordered sets. Other
similar usages will not be found and are false negatives.
