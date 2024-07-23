// RUN: %check_clang_tidy -std=c++14 %s bugprone-incorrect-iterators %t

namespace std {
namespace execution {
class parallel_policy {};
constexpr parallel_policy par;
} // namespace execution

template <typename BiDirIter> class reverse_iterator {
  constexpr explicit reverse_iterator(BiDirIter Iter);
};

template <typename BiDirIter>
reverse_iterator<BiDirIter> make_reverse_iterator(BiDirIter Iter);

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using reverse_const_iterator = std::reverse_iterator<const_iterator>;

  constexpr const_iterator begin() const;
  constexpr const_iterator end() const;
  constexpr const_iterator cbegin() const;
  constexpr const_iterator cend() const;
  constexpr iterator begin();
  constexpr iterator end();
  constexpr reverse_const_iterator rbegin() const;
  constexpr reverse_const_iterator rend() const;
  constexpr reverse_const_iterator crbegin() const;
  constexpr reverse_const_iterator crend() const;
  constexpr reverse_iterator rbegin();
  constexpr reverse_iterator rend();

  template <class InputIt>
  iterator insert(const_iterator pos, InputIt first, InputIt last);
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

template <typename Container> constexpr auto rbegin(const Container &Cont) {
  return Cont.rbegin();
}

template <typename Container> constexpr auto rbegin(Container &Cont) {
  return Cont.rbegin();
}

template <typename Container> constexpr auto rend(const Container &Cont) {
  return Cont.rend();
}

template <typename Container> constexpr auto rend(Container &Cont) {
  return Cont.rend();
}

template <typename Container> constexpr auto crbegin(const Container &Cont) {
  return Cont.crbegin();
}

template <typename Container> constexpr auto crend(const Container &Cont) {
  return Cont.crend();
}
// Find
template <class InputIt, class T>
InputIt find(InputIt first, InputIt last, const T &value);

template <class Policy, class InputIt, class T>
InputIt find(Policy &&policy, InputIt first, InputIt last, const T &value);

template <class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   InputIt2 last2, OutputIt d_first);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class ForwardIt3>
ForwardIt3 set_union(ExecutionPolicy &&policy, ForwardIt1 first1,
                     ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2,
                     ForwardIt3 d_first);

template <class RandomIt> void push_heap(RandomIt first, RandomIt last);

template <class BidirIt1, class BidirIt2>
BidirIt2 copy_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);

template <class ForwardIt1, class ForwardIt2>
ForwardIt1 find_end(ForwardIt1 first, ForwardIt1 last, ForwardIt2 s_first,
                    ForwardIt2 s_last);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
ForwardIt1 find_end(ExecutionPolicy &&policy, ForwardIt1 first, ForwardIt1 last,
                    ForwardIt2 s_first, ForwardIt2 s_last);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
bool equal(ExecutionPolicy &&policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
bool equal(ExecutionPolicy &&policy, ForwardIt1 first1, ForwardIt1 last1,
           ForwardIt2 first2, ForwardIt2 last2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);

template <class ForwardIt1, class ForwardIt2>
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                    ForwardIt2 last2);

template <class InputIt, class NoThrowForwardIt>
NoThrowForwardIt uninitialized_copy(InputIt first, InputIt last,
                                    NoThrowForwardIt d_first);

template <class ExecutionPolicy, class ForwardIt, class NoThrowForwardIt>
NoThrowForwardIt uninitialized_copy(ExecutionPolicy &&policy, ForwardIt first,
                                    ForwardIt last, NoThrowForwardIt d_first);

template <class InputIt, class OutputIt>
OutputIt partial_sum(InputIt first, InputIt last, OutputIt d_first);

} // namespace std

void Test() {
  std::vector<int> I;
  std::vector<int> J;
  std::vector<int> K;
  std::find(I.end(), I.begin(), 0);
  // CHECK-NOTES: :[[@LINE-1]]:13: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:22: warning: 'begin' iterator supplied where an 'end' iterator is expected
  std::find(std::execution::par, I.begin(), J.end(), 0);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:34: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:45: note: different range passed here to end
  std::find(std::make_reverse_iterator(I.end()),
            std::make_reverse_iterator(I.end()), 0);
  // CHECK-NOTES: :[[@LINE-1]]:40: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:13: note: 'make_reverse_iterator<int *>' changes 'end' into a 'begin' iterator
  std::find(I.rbegin(), std::make_reverse_iterator(J.begin()), 0);
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:13: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:52: note: different range passed here to end

  I.insert(J.begin(), J.end(), J.begin());
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: 'insert<int *>' called with an iterator for a different container
  // CHECK-NOTES: :[[@LINE-2]]:3: note: container is specified here
  // CHECK-NOTES: :[[@LINE-3]]:12: note: different container provided here
  // CHECK-NOTES: :[[@LINE-4]]:23: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-5]]:32: warning: 'begin' iterator supplied where an 'end' iterator is expected

  std::set_union(I.begin(), I.begin(), J.end(), J.end(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:40: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:58: warning: 'end' iterator supplied where an output iterator is expected
  std::set_union(std::execution::par, I.begin(), I.begin(), J.end(), J.end(),
                 K.end());
  // CHECK-NOTES: :[[@LINE-2]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:61: warning: 'end' iterator supplied where a 'begin' iterator is expected
  // CHECK-NOTES: :[[@LINE-3]]:18: warning: 'end' iterator supplied where an output iterator is expected

  std::push_heap(std::begin(I), std::end(J));
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:29: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:42: note: different range passed here to end

  std::copy_backward(I.begin(), J.end(), K.begin());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:22: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:33: note: different range passed here to end
  // CHECK-NOTES: :[[@LINE-4]]:42: warning: 'begin' iterator supplied where an 'end' iterator is expected

  std::find_end(I.rbegin(), I.rbegin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:29: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:41: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::find_end(std::execution::par, I.rbegin(), I.rbegin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:50: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:62: warning: 'end' iterator supplied where a 'begin' iterator is expected

  std::equal(I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:34: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::equal(std::execution::par, I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:55: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::equal(I.begin(), I.end(), J.rbegin(), K.rend());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:34: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:46: note: different range passed here to end
  std::equal(std::execution::par, I.begin(), I.end(), J.begin(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:55: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:66: note: different range passed here to end

  std::is_permutation(I.begin(), J.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:23: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:34: note: different range passed here to end
  // CHECK-NOTES: :[[@LINE-4]]:43: warning: 'end' iterator supplied where a 'begin' iterator is expected
  std::is_permutation(I.begin(), I.end(), J.begin(), K.end());
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: mismatched ranges pass to function
  // CHECK-NOTES: :[[@LINE-2]]:43: note: first range passed here to begin
  // CHECK-NOTES: :[[@LINE-3]]:54: note: different range passed here to end

  std::uninitialized_copy(I.begin(), I.end(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:47: warning: 'end' iterator supplied where an output iterator is expected
  std::uninitialized_copy(std::execution::par, I.begin(), I.begin(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:59: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:70: warning: 'end' iterator supplied where an output iterator is expected

  std::partial_sum(I.begin(), I.begin(), J.end());
  // CHECK-NOTES: :[[@LINE-1]]:31: warning: 'begin' iterator supplied where an 'end' iterator is expected
  // CHECK-NOTES: :[[@LINE-2]]:42: warning: 'end' iterator supplied where an output iterator is expected

}
