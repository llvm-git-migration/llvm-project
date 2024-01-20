// RUN: %clang -x cl -c %s
// RUN: %clang -x cl --save-temps -c %s

uint3 add(uint3 a, uint3 b) {
  ulong x = a.x + (ulong)b.x;
  ulong y = a.y + (ulong)b.y + (x >> 32);
  uint z = a.z + b.z + (y >> 32);
  return (uint3)(x, y, z);
}
