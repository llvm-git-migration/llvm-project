//===- DialectLLVM.cpp - Pybind module for LLVM dialect API support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <string>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

namespace {
/// Standalone RAII scope guard. We don't want to depend on the LLVM support
/// library here to simplify build.
template <typename FnTy>
class OnScopeExit {
public:
  OnScopeExit(FnTy &&fn) : callback(std::forward<FnTy>(fn)) {}
  ~OnScopeExit() { callback(); }

private:
  FnTy callback;
};

template <typename FnTy>
OnScopeExit(FnTy &&fn) -> OnScopeExit<FnTy>;
} // namespace

void populateDialectLLVMSubmodule(const pybind11::module &m) {
  auto llvmStructType =
      mlir_type_subclass(m, "StructType", mlirTypeIsALLVMStructType);

  llvmStructType.def_classmethod(
      "get_literal",
      [](py::object cls, const std::vector<MlirType> &elements, bool packed,
         MlirLocation loc) {
        std::string errorMessage = "";
        auto handler = +[](MlirDiagnostic diag, void *data) {
          auto printer = +[](MlirStringRef message, void *data) {
            *static_cast<std::string *>(data) +=
                StringRef(message.data, message.length);
          };
          mlirDiagnosticPrint(diag, printer, data);
          return mlirLogicalResultSuccess();
        };

        MlirContext context = mlirLocationGetContext(loc);
        MlirDiagnosticHandlerID diagID = mlirContextAttachDiagnosticHandler(
            context, handler, &errorMessage, nullptr);
        OnScopeExit scopeGuard(
            [&]() { mlirContextDetachDiagnosticHandler(context, diagID); });

        MlirType type = mlirLLVMStructTypeLiteralGetChecked(
            loc, elements.size(), elements.data(), packed);
        if (mlirTypeIsNull(type)) {
          throw py::value_error(errorMessage);
        }
        return cls(type);
      },
      py::arg("cls"), py::arg("elements"), py::kw_only(),
      py::arg("packed") = false, py::arg("loc") = py::none());

  llvmStructType.def_classmethod(
      "get_identified",
      [](py::object cls, const std::string &name, MlirContext context) {
        return cls(mlirLLVMStructTypeIdentifiedGet(
            context, mlirStringRefCreate(name.data(), name.size())));
      },
      py::arg("cls"), py::arg("name"), py::kw_only(),
      py::arg("context") = py::none());

  llvmStructType.def_classmethod(
      "get_opaque",
      [](py::object cls, const std::string &name, MlirContext context) {
        return cls(mlirLLVMStructTypeOpaqueGet(
            context, mlirStringRefCreate(name.data(), name.size())));
      }, py::arg("cls"), py::arg("name"), py::arg("context") = py::none());

  llvmStructType.def(
      "set_body",
      [](MlirType self, const std::vector<MlirType> &elements, bool packed) {
        MlirLogicalResult result = mlirLLVMStructTypeSetBody(
            self, elements.size(), elements.data(), packed);
        if (!mlirLogicalResultIsSuccess(result)) {
          throw py::value_error(
              "Struct body already set to different content.");
        }
      },
      py::arg("elements"), py::kw_only(), py::arg("packed") = false);

  llvmStructType.def_classmethod(
      "new_identified",
      [](py::object cls, const std::string &name,
         const std::vector<MlirType> &elements, bool packed, MlirContext ctx) {
        return cls(mlirLLVMStructTypeIdentifiedNewGet(
            ctx, mlirStringRefCreate(name.data(), name.length()),
            elements.size(), elements.data(), packed));
      },
      py::arg("cls"), py::arg("name"), py::arg("elements"), py::kw_only(),
      py::arg("packed") = false, py::arg("context") = py::none());

  llvmStructType.def_property_readonly(
      "name", [](MlirType type) -> std::optional<std::string> {
        if (mlirLLVMStructTypeIsLiteral(type))
          return std::nullopt;

        MlirStringRef stringRef = mlirLLVMStructTypeGetIdentifier(type);
        return StringRef(stringRef.data, stringRef.length).str();
      });

  llvmStructType.def_property_readonly("body", [](MlirType type) -> py::object {
    // Don't crash in absence of a body.
    if (mlirLLVMStructTypeIsOpaque(type))
      return py::none();

    py::list body;
    for (intptr_t i = 0, e = mlirLLVMStructTypeGetNumElementTypes(type); i < e;
         ++i) {
      body.append(mlirLLVMStructTypeGetElementType(type, i));
    }
    return body;
  });

  llvmStructType.def_property_readonly(
      "packed", [](MlirType type) { return mlirLLVMStructTypeIsPacked(type); });

  llvmStructType.def_property_readonly(
      "opaque", [](MlirType type) { return mlirLLVMStructTypeIsOpaque(type); });
}

PYBIND11_MODULE(_mlirDialectsLLVM, m) {
  m.doc() = "MLIR LLVM Dialect";

  populateDialectLLVMSubmodule(m);
}