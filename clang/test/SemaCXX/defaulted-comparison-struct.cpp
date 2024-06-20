// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template <typename> class a {};
template <typename b> b c(a<b>);
template <typename d> class e {
public:
  typedef a<d *> f;
  f begin();
};
template <typename d, typename g> constexpr bool operator==(d h, g i) {
  return *c(h.begin()) == *c(i.begin());
}
struct j {
  e<j> bar;
  bool operator==(const j &) const;
};
bool j::operator==(const j &) const = default;
