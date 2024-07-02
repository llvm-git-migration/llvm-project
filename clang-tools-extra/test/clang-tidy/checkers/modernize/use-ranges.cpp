// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t

// CHECK-FIXES: #include <algorithm>

namespace std {

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  constexpr const_iterator begin() const;
  constexpr const_iterator end() const;
  constexpr const_iterator cbegin() const;
  constexpr const_iterator cend() const;
  constexpr iterator begin();
  constexpr iterator end();
};

template <typename Container> constexpr auto begin(const Container &Cont) {
  return Cont.begin();
}

template <typename Container> constexpr auto begin(Container &Cont) {
  return Cont.begin();
}

template <typename Container> constexpr auto end(const Container &Cont) {
  return Cont.end();
}

template <typename Container> constexpr auto end(Container &Cont) {
  return Cont.end();
}

template <typename Container> constexpr auto cbegin(const Container &Cont) {
  return Cont.cbegin();
}

template <typename Container> constexpr auto cend(const Container &Cont) {
  return Cont.cend();
}
// Find
template< class InputIt, class T >
InputIt find( InputIt first, InputIt last, const T& value );

// Reverse
template <typename Iter> void reverse(Iter begin, Iter end);

template <typename Iter>
void reverse(int policy, Iter begin, Iter end);

// Includes
template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);
template <class ForwardIt1, class ForwardIt2>
bool includes(int policy, ForwardIt1 first1, ForwardIt1 last1,
              ForwardIt2 first2, ForwardIt2 last2);

// IsPermutation
template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);
template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

// Equal
template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

template <class ForwardIt1, class ForwardIt2>
bool equal(int policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class ForwardIt1, class ForwardIt2>
bool equal(int policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2);

template <class InputIt1, class InputIt2, class BinaryPred>
bool equal(InputIt1 first1, InputIt1 last1,
           InputIt2 first2, InputIt2 last2, BinaryPred p) {
  // Need a definition to suppress undefined_internal_type when invoked with lambda
  return true;
}

} // namespace std

void Positives() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 0);

  std::find(I.cbegin(), I.cend(), 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 1);

  std::find(std::begin(I), std::end(I), 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 2);

  std::find(std::cbegin(I), std::cend(I), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 3);

  std::find(std::cbegin(I), I.cend(), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 4);

  std::reverse(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::reverse(I);

  std::reverse(0, I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::reverse(0, I);

  std::includes(I.begin(), I.end(), I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, I);

  std::includes(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, J);


  std::includes(0, I.begin(), I.end(), I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(0, I, I);

  std::includes(0, I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(0, I, J);

  std::is_permutation(I.begin(), I.end(), J.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::is_permutation(I, J.begin());

  std::is_permutation(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::is_permutation(I, J);

  std::equal(I.begin(), I.end(), J.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J.begin());

  std::equal(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J);

  std::equal(0, I.begin(), I.end(), J.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(0, I, J.begin());

  std::equal(1, I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(1, I, J);

  std::equal(I.begin(), I.end(), J.end(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J.end(), J.end());

  std::equal(I.begin(), I.end(), J.end(), J.end(), [](int a, int b){ return a == b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J.end(), J.end(), [](int a, int b){ return a == b; });


  using std::find;

  find(I.begin(), I.end(), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 5);
}

void Negatives() {
  std::vector<int> I, J;
  std::find(I.begin(), J.end(), 0);
  std::find(I.begin(), I.begin(), 0);
  std::find(I.end(), I.begin(), 0);
  std::equal(I.begin(), J.end(), I.begin(), I.end());
}
