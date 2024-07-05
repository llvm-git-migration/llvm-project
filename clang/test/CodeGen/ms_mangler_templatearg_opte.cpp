// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc19.33.0 -emit-llvm -std=c++20 -x c++ < %s | FileCheck --check-prefix=WIN64 %s

/*

this works (linux host):
clang++ -std=c++20 -c ms_mangler_templatearg_opte.cpp -o /dev/null

this dost not:
/usr/local/bin/clang-17 -cc1 -triple x86_64-pc-windows-msvc19.33.0 -emit-obj -std=c++20 -o ms_mangler_templatearg_opte.obj -x c++ ms_mangler_templatearg_opte.cpp

The first pass is fine, it's the final pass of cesum where L.data = (&ints)+1 that clang bawks at. Obviously this address can't be dereferenced, but the `if constexpr` sees to that. The unused template param should not break the mangler.

*/


typedef long long unsigned size_t;

template<class T> struct llist {
  const T* data;
  size_t len;
  constexpr llist(const T* data, size_t len) : data(data), len(len) {};
  constexpr inline bool empty() const { return len == 0; };
  constexpr llist<T> next() const { return { data+1, len-1 }; };
  constexpr const T& peek() const { return data[0]; };
};

//recurse to iterate over the list, without the need for a terminal overload or duplicated handling of the terminal case
template<llist<int> L> int cesum() {
  if constexpr(L.empty()) {
    return 0;
  } else {
    return L.peek() + cesum<L.next()>();
  }
};

//constexpr int ints[] = { 1, 2, 7, 8, 9, -17, -10 }; //Note: this does NOT break the unpatched mangler
constexpr int ints = 7;

int main() {
  return cesum<llist<int>(&ints, 1)>();//taking address of non-array
};

