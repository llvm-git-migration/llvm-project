// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

using size_t = __typeof(sizeof(int));
namespace std{
template <class T, size_t N>
class array {
  T elements[N];
  
  public:
  T& operator[](unsigned i) { return elements[i]; }
  constexpr const T* data() const noexcept {
     return elements;
  }

};
}

class ArrayClass {
public:
    typedef std::array<std::array<double, 4>, 4> Matrix;
    double e() { return matrix[3][0]; }
    Matrix matrix;
};

class AnotherClass {
    Ref<ArrayClass> matrix;
    void test() {
      double val[] = { matrix->e(), matrix->e() };
    }
};
