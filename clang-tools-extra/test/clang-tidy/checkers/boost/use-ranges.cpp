// RUN: %check_clang_tidy -std=c++14 %s boost-use-ranges %t

// CHECK-FIXES: #include <boost/range/algorithm/find.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/reverse.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/set_algorithm.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/equal.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/permutation.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/heap_algorithm.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/copy_if.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/is_sorted.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx17/reduce.hpp>

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
InputIt find(InputIt first, InputIt last, const T& value);

template <typename Iter> void reverse(Iter begin, Iter end);

template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

template <class BidirIt>
bool next_permutation(BidirIt first, BidirIt last);

template <class ForwardIt1, class ForwardIt2>
bool equal(ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2);

template <class RandomIt>
void push_heap(RandomIt first, RandomIt last);

template <class InputIt, class OutputIt, class UnaryPred>
OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, UnaryPred pred);

template <class ForwardIt>
ForwardIt is_sorted_until(ForwardIt first, ForwardIt last);

template <class InputIt>
void reduce(InputIt first, InputIt last);

template< class InputIt, class T >
T reduce(InputIt first, InputIt last, T init);

} // namespace std

bool return_true(int val) {
  return true;
}

void Positives() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::find(I, 0);

  std::reverse(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::reverse(I);

  std::includes(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::includes(I, J);

  std::equal(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::equal(I, J);

  std::next_permutation(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::next_permutation(I);

  std::push_heap(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::range::push_heap(I);

  std::copy_if(I.begin(), I.end(), J.begin(), &return_true);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::algorithm::copy_if(I, J.begin(), &return_true);

  std::is_sorted_until(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::algorithm::is_sorted_until(I);

  std::reduce(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I);

  std::reduce(I.begin(), I.end(), 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Use a ranges version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 2);
}
