//===-- Lower/OpenMP/Utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPUTILS_H
#define FORTRAN_LOWER_OPENMPUTILS_H

namespace Fortran {

namespace semantics {
class Symbol;
} // namespace semantics

namespace parser {
struct OmpObject;
} // namespace parser

namespace lower {
namespace omp {

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
