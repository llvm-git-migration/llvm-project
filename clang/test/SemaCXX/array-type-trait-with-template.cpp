// RUN: %clang_cc1 -fsyntax-only %s

// When __array_rank is used with a template type parameter, this
// test ensures clang considers the final expression as having an
// integral type.
//
// Although array_extent was handled well, it is added here.
template <typename T, int N>
constexpr int array_rank(T (&lhs)[N]) {
  return __array_rank(T[N]);
}

template <int I, typename T, int N>
constexpr int array_extent(T (&lhs)[N]) {
  return __array_extent(T[N], I);
}

int main() {
  constexpr int vec[] = {0, 1, 2, 1};
  constexpr int mat[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
  };

  (void) (array_rank(vec) == 1);
  (void) (array_rank(vec) == 2);

  static_assert(array_rank(vec) == 1);
  static_assert(array_rank(mat) == 2);

  static_assert(array_extent<0>(vec) == 4);
  static_assert(array_extent<0>(mat) == 4);
  static_assert(array_extent<1>(mat) == 4);
  static_assert(array_extent<1>(vec) == 0);
}
