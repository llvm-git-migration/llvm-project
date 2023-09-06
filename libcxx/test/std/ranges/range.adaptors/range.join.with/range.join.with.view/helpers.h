//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_RANGE_JOIN_WITH_VIEW_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_RANGE_JOIN_WITH_VIEW_H

#include <concepts>
#include <utility>

template <class T>
void pass(T);

template <class T, class... Args>
concept ConstructionIsExplicit =
    std::constructible_from<T, Args...> && !requires(Args&&... args) { pass<T>({std::forward<Args>(args)...}); };

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_WITH_RANGE_JOIN_WITH_VIEW_H
