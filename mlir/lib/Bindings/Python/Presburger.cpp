#include "mlir-c/Presburger.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "llvm/ADT/ScopeExit.h"
#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

static bool isSignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
         code == 'q';
}

namespace {
struct PyPresburgerIntegerRelation {
  PyPresburgerIntegerRelation(MlirPresburgerIntegerRelation relation)
      : relation(relation) {}
  PyPresburgerIntegerRelation(PyPresburgerIntegerRelation &&other) noexcept
      : relation(other.relation) {
    other.relation.ptr = nullptr;
  }
  ~PyPresburgerIntegerRelation() {
    if (relation.ptr) {
      mlirPresburgerIntegerRelationDestroy(relation);
      relation.ptr = {nullptr};
    }
  }
  static PyPresburgerIntegerRelation
  getFromBuffers(py::buffer inequalitiesCoefficients,
                 py::buffer equalityCoefficients, unsigned numDomainVars,
                 unsigned numRangeVars);
  py::object getCapsule();
  static void bind(py::module &module);
  MlirPresburgerIntegerRelation relation{nullptr};
};

/// A utility that enables accessing/modifying the underlying coefficients
/// easier.
struct PyPresburgerTableau {
  enum class Kind { Equalities, Inequalities };
  PyPresburgerTableau(MlirPresburgerIntegerRelation relation, Kind kind)
      : relation(relation), kind(kind) {}
  static void bind(py::module &module);
  int64_t at64(int64_t row, int64_t col) const {
    if (kind == Kind::Equalities)
      return mlirPresburgerIntegerRelationAtEq64(relation, row, col);
    return mlirPresburgerIntegerRelationAtIneq64(relation, row, col);
  }
  MlirPresburgerIntegerRelation relation;
  Kind kind;
};
} // namespace

PyPresburgerIntegerRelation PyPresburgerIntegerRelation::getFromBuffers(
    py::buffer inequalitiesCoefficients, py::buffer equalityCoefficients,
    unsigned numDomainVars, unsigned numRangeVars) {
  // Request a contiguous view. In exotic cases, this will cause a copy.
  int flags = PyBUF_ND;
  flags |= PyBUF_FORMAT;
  // Get the view of the inequality coefficients.
  std::unique_ptr<Py_buffer> ineqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(inequalitiesCoefficients.ptr(), ineqView.get(),
                         flags) != 0)
    throw py::error_already_set();
  auto freeIneqBuffer = llvm::make_scope_exit([&]() {
    if (ineqView)
      PyBuffer_Release(ineqView.get());
  });
  if (!PyBuffer_IsContiguous(ineqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");
  if (!isSignedIntegerFormat(ineqView->format) || ineqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(ineqView->format));
  if (ineqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d inequality coefficients but got rank ") +
        std::to_string(ineqView->ndim));
  unsigned numInequalities = ineqView->shape[0];
  // Get the view of the eequality coefficients.
  std::unique_ptr<Py_buffer> eqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(equalityCoefficients.ptr(), eqView.get(), flags) != 0)
    throw py::error_already_set();
  auto freeEqBuffer = llvm::make_scope_exit([&]() {
    if (eqView)
      PyBuffer_Release(eqView.get());
  });
  if (!PyBuffer_IsContiguous(eqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");
  if (!isSignedIntegerFormat(eqView->format) || eqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(eqView->format));
  if (eqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d equality coefficients but got rank ") +
        std::to_string(eqView->ndim));
  unsigned numEqualities = eqView->shape[0];
  if (eqView->shape[1] != numDomainVars + numRangeVars + 1 ||
      eqView->shape[1] != ineqView->shape[1])
    throw std::invalid_argument(
        "expected number of columns of inequality and equality coefficient "
        "matrices to equal numRangeVars + numDomainVars + 1");
  MlirPresburgerIntegerRelation relation =
      mlirPresburgerIntegerRelationCreateFromCoefficients(
          reinterpret_cast<const int64_t *>(ineqView->buf), numInequalities,
          reinterpret_cast<const int64_t *>(eqView->buf), numEqualities,
          numDomainVars, numRangeVars);
  return PyPresburgerIntegerRelation(relation);
}

py::object PyPresburgerIntegerRelation::getCapsule() {
  throw std::invalid_argument("unimplemented");
}

void PyPresburgerTableau::bind(py::module &m) {
  py::class_<PyPresburgerTableau>(m, "IntegerRelationTableau",
                                  py::module_local())
      .def("__getitem__", [](PyPresburgerTableau &self,
                             const py::tuple &index) {
        return self.at64(index[0].cast<int64_t>(), index[1].cast<int64_t>());
      });
}

void PyPresburgerIntegerRelation::bind(py::module &m) {
  py::class_<PyPresburgerIntegerRelation>(m, "IntegerRelation",
                                          py::module_local())
      .def(py::init<>(&PyPresburgerIntegerRelation::getFromBuffers))
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyPresburgerIntegerRelation::getCapsule)
      .def("__eq__",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationIsEqual(self.relation,
                                                         other.relation);
           })
      .def("append",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationAppend(self.relation,
                                                        other.relation);
           })
      .def("intersect",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             PyPresburgerIntegerRelation intersection(
                 mlirPresburgerIntegerRelationIntersect(self.relation,
                                                        other.relation));
             return intersection;
           })
      .def("is_equal",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationIsEqual(self.relation,
                                                         other.relation);
           })
      .def("is_obviously_equal",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationIsObviouslyEqual(
                 self.relation, other.relation);
           })
      .def("is_subset_of",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationIsSubsetOf(self.relation,
                                                            other.relation);
           })
      .def(
          "inequalities",
          [](PyPresburgerIntegerRelation &self) {
            PyPresburgerTableau tableau(
                self.relation, PyPresburgerTableau::Kind::Inequalities);
            return tableau;
          },
          py::keep_alive<0, 1>())
      .def(
          "equalities",
          [](PyPresburgerIntegerRelation &self) {
            PyPresburgerTableau tableau(self.relation,
                                        PyPresburgerTableau::Kind::Equalities);
            return tableau;
          },
          py::keep_alive<0, 1>())
      .def("get_equality",
           [](PyPresburgerIntegerRelation &self, int64_t row) {
             unsigned numCol =
                 mlirPresburgerIntegerRelationNumCols(self.relation);
             std::vector<int64_t> result(numCol);
             for (unsigned i = 0; i < numCol; i++)
               result[i] =
                   mlirPresburgerIntegerRelationAtEq64(self.relation, row, i);
             return result;
           })
      .def("get_inequality",
           [](PyPresburgerIntegerRelation &self, int64_t row) {
             unsigned numCol =
                 mlirPresburgerIntegerRelationNumCols(self.relation);
             std::vector<int64_t> result(numCol);
             for (unsigned i = 0; i < numCol; i++)
               result[i] =
                   mlirPresburgerIntegerRelationAtIneq64(self.relation, row, i);
             return result;
           })
      .def_property_readonly(
          "num_constraints",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumConstraints(self.relation);
          })
      .def_property_readonly(
          "num_domain_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumDomainVars(self.relation);
          })
      .def_property_readonly("num_range_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumRangeVars(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_symbol_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumSymbolVars(self.relation);
          })
      .def_property_readonly("num_local_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumLocalVars(
                                   self.relation);
                             })
      .def_property_readonly("num_dim_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumDimVars(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_dim_and_symbol_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumDimAndSymbolVars(
                self.relation);
          })
      .def_property_readonly("num_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumVars(
                                   self.relation);
                             })
      .def_property_readonly("num_columns",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumCols(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_equalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumEqualities(self.relation);
          })
      .def_property_readonly(
          "num_inequalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumInequalities(self.relation);
          })
      .def_property_readonly(
          "num_reserved_equalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumReservedEqualities(
                self.relation);
          })
      .def_property_readonly(
          "num_reserved_inequalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumReservedInequalities(
                self.relation);
          })
      .def("__str__", [](const PyPresburgerIntegerRelation &self) {
        mlirPresburgerIntegerRelationDump(self.relation);
        return "";
      });
}

static void populatePresburgerModule(py::module &m) {
  PyPresburgerTableau::bind(m);
  PyPresburgerIntegerRelation::bind(m);
}
// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------
PYBIND11_MODULE(_mlirPresburger, m) {
  m.doc() = "MLIR Presburger utilities";
  populatePresburgerModule(m);
}