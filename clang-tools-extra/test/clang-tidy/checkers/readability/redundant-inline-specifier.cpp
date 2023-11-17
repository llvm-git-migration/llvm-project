// RUN: %check_clang_tidy %s readability-redundant-inline-specifier %t

template <typename T> inline T f()
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: Function 'f' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: template <typename T> T f()
{
    return T{};
}

template <> inline double f<double>() = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: Function 'f<double>' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: template <> double f<double>() = delete;

inline int g(float a)
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:1: warning: Function 'g' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
{
    return static_cast<int>(a - 5.F);
}

inline int g(double) = delete;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: Function 'g' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: int g(double) = delete;

class C
{
  public:
    inline C& operator=(const C&) = delete;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Function 'operator=' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: C& operator=(const C&) = delete;

    constexpr inline C& operator=(int a);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Function 'operator=' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr C& operator=(int a);

    inline C() {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Function 'C' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: C() {}

    constexpr inline C(int);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: Function 'C' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr C(int);

    inline int Get42() const { return 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Function 'Get42' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: int Get42() const { return 42; }

    static inline constexpr int C_STATIC = 42;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: Variable 'C_STATIC' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: static constexpr int C_STATIC = 42;

    static constexpr int C_STATIC_2 = 42;
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: Variable 'C_STATIC_2' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
};

constexpr inline int Get42() { return 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: Function 'Get42' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: constexpr int Get42() { return 42; }


static constexpr inline int NAMESPACE_STATIC = 42;
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:18: warning: Variable 'NAMESPACE_STATIC' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]

inline static int fn0(int i)
// CHECK-MESSAGES-NOT: :[[@LINE-1]]:1: warning: Function 'fn0' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
{
    return i - 1;
}

static constexpr inline int fn1(int i)
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: Function 'fn1' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
// CHECK-FIXES: static constexpr int fn1(int i)
{
    return i - 1;
}

namespace
{
    inline int fn2(int i)
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: Function 'fn2' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    {
        return i - 1;
    }

    inline constexpr int fn3(int i)
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Function 'fn3' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr int fn3(int i)
    {
        return i - 1;
    }
}

namespace ns
{
    inline int fn4(int i)
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: Function 'fn4' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    {
        return i - 1;
    }

    inline constexpr int fn5(int i)
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: Function 'fn5' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]
    // CHECK-FIXES: constexpr int fn5(int i)
    {
        return i - 1;
    }
}

auto fn6 = [](){};
//CHECK-MESSAGES-NOT: :[[@LINE-1]]:1: warning: Function 'operator()' has inline specifier but is implicitly inlined [readability-redundant-inline-specifier]

