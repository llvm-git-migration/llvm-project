// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-bit-cast-pointers %t

namespace std
{
template <typename To, typename From>
To bit_cast(From from)
{
  // Dummy implementation for the purpose of the test.
  // We don't want to include <cstring> to get std::memcpy.
  To to{};
  return to;
}
}

void pointer2pointer()
{
  int x{};
  float bad = *std::bit_cast<float*>(&x); // UB, but looks safe due to std::bit_cast
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use std::bit_cast on pointers; use it on values instead [bugprone-bit-cast-pointers]
  float good = std::bit_cast<float>(x);   // Well-defined
}

// Pointer-integer conversions are allowed by this check
void int2pointer()
{
  unsigned long long addr{};
  float* p = std::bit_cast<float*>(addr);
}

void pointer2int()
{
  float* p{};
  auto addr = std::bit_cast<unsigned long long>(p);
}
