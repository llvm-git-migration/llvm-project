// RUN: %clang_cc1 -emit-llvm -std=c++20 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

// COMDAT groups in ELF objects are not permitted to contain local symbols. While template parameter
// objects are normally emitted in COMDATs, we shouldn't do so if they would have internal linkage.

extern "C" int printf(...);
typedef __typeof__(sizeof(0)) size_t;

namespace {
template<size_t N>
struct DebugContext
{
    char value[N];
    constexpr DebugContext(const char (&str)[N]) {
        for (size_t i = 0; i < N; ++i) {
            value[i] = str[i];
        }
    }
};
}

template<DebugContext Context>
struct ConditionalDebug
{
    public:
    static void log() {
        printf("%s", Context.value);
    }
};

using Debug = ConditionalDebug<"compartment A">;

void foo() {
	Debug::log();
}

// CHECK-NOT: $_ZTAXtlN12_GLOBAL__N_112DebugContextILm14EEEtlA14_cLc99ELc111ELc109ELc112ELc97ELc114ELc116ELc109ELc101ELc110ELc116ELc32ELc65EEEE = comdat any
// CHECK: @_ZTAXtlN12_GLOBAL__N_112DebugContextILm14EEEtlA14_cLc99ELc111ELc109ELc112ELc97ELc114ELc116ELc109ELc101ELc110ELc116ELc32ELc65EEEE = internal constant %"struct.(anonymous namespace)::DebugContext" { [14 x i8] c"compartment A\00" }