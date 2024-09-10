// RUN: llvm-mc -triple bpfel --mcpu=v5 --assemble --filetype=obj %s \
// RUN:   | llvm-objdump -d - | FileCheck %s

// CHECK: f1 10 00 00 00 00 00 00	w0 = load_acquire((u8 *)(r1 + 0x0))
// CHECK: e9 10 00 00 00 00 00 00	w0 = load_acquire((u16 *)(r1 + 0x0))
// CHECK: e1 10 00 00 00 00 00 00	w0 = load_acquire((u32 *)(r1 + 0x0))
w0 = load_acquire((u8 *)(r1 + 0))
w0 = load_acquire((u16 *)(r1 + 0))
w0 = load_acquire((u32 *)(r1 + 0))

// CHECK: f9 10 00 00 00 00 00 00	r0 = load_acquire((u64 *)(r1 + 0x0))
r0 = load_acquire((u64 *)(r1 + 0))

// CHECK: f3 21 00 00 00 00 00 00	store_release((u8 *)(r1 + 0x0), w2)
// CHECK: eb 21 00 00 00 00 00 00	store_release((u16 *)(r1 + 0x0), w2)
// CHECK: e3 21 00 00 00 00 00 00	store_release((u32 *)(r1 + 0x0), w2)
store_release((u8 *)(r1 + 0), w2)
store_release((u16 *)(r1 + 0), w2)
store_release((u32 *)(r1 + 0), w2)

// CHECK: fb 21 00 00 00 00 00 00	store_release((u64 *)(r1 + 0x0), r2)
store_release((u64 *)(r1 + 0), r2)
