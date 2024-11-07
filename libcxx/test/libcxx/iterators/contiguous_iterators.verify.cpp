//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <iterator>

// __bounded_iter<_Iter>
// __wrap_iter<_Iter>

// Verify that these wrappers do not accept non-contiguous iterators as determined by
// __libcpp_is_contiguous_iterator.
// static_assert should be used, see https://github.com/llvm/llvm-project/issues/115002.

#include <deque>
#include <iterator>

// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Only contiguous iterators can be adapted by __bounded_iter.}}
std::__bounded_iter<std::deque<int>::iterator> bit;
// expected-error-re@*:* {{static assertion failed due to requirement {{.*}}Only contiguous iterators can be adapted by __wrap_iter.}}
std::__wrap_iter<std::deque<int>::iterator> wit;
