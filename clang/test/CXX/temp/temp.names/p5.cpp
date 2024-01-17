// RUN: %clang_cc1 -fsyntax-only -pedantic-errors -verify %s

template<typename T> struct A {
  template<typename U> struct B {
    struct C;
    template<typename V> struct D;
    template<typename V> struct D<V*>;

    void f();
    template<typename V> void g();

    static int x;
    template<typename V> static int y;
    template<typename V> static int y<V*>;

    enum class E;
  };
};

template<typename T>
template<typename U>
struct A<T>::template B<U>::C { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
struct A<int>::template B<bool>::C; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
struct A<int>::template B<bool>::C { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::template B<U>::D { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::template B<U>::D<V*> { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D<V*> { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
void A<T>::template B<U>::f() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
void A<int>::template B<bool>::f() { } // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
void A<T>::template B<U>::g() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
void A<int>::template B<bool>::g() { } // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
int A<T>::template B<U>::x = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
int A<T>::template B<U>::y = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
int A<T>::template B<U>::y<V*> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
enum class A<T>::template B<U>::E { a }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
enum class A<int>::template B<bool>::E; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
enum class A<int>::template B<bool>::E { a }; // expected-error{{'template' cannot be used after a declarative}}
