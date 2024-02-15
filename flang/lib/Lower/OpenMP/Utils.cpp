//===-- Utils..cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include <flang/Parser/parse-tree.h>
#include <flang/Parser/tools.h>
#include <flang/Semantics/tools.h>
#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> treatIndexAsSection(
    "openmp-treat-index-as-section",
    llvm::cl::desc("In the OpenMP data clauses treat `a(N)` as `a(N:N)`."),
    llvm::cl::init(true));

namespace Fortran {
namespace lower {
namespace omp {

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject) {
  Fortran::semantics::Symbol *sym = nullptr;
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto *arrayEle =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator)) {
              sym = GetFirstName(arrayEle->base).symbol;
            } else if (auto *structComp = Fortran::parser::Unwrap<
                           Fortran::parser::StructureComponent>(designator)) {
              sym = structComp->component.symbol;
            } else if (const Fortran::parser::Name *name =
                           Fortran::semantics::getDesignatorNameIfDataRef(
                               designator)) {
              sym = name->symbol;
            }
          },
          [&](const Fortran::parser::Name &name) { sym = name.symbol; }},
      ompObject.u);
  return sym;
}

} // namespace omp
} // namespace lower
} // namespace Fortran
