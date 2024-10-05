; RUN: llc < %s -march=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding \
; RUN:   | FileCheck -check-prefixes=CHECK-LE %s
; RUN: llc < %s -march=bpfeb -mcpu=v4 -verify-machineinstrs -show-mc-encoding \
; RUN:   | FileCheck -check-prefixes=CHECK-BE %s

define dso_local i8 @load_acquire_i8(ptr nocapture noundef readonly %p) local_unnamed_addr {
; CHECK-LABEL: load_acquire_i8
; CHECK-LE: w0 = load_acquire((u8 *)(r1 + 0)) # encoding: [0xd3,0x10,0x00,0x00,0x12,0x00,0x00,0x00]
; CHECK-BE: w0 = load_acquire((u8 *)(r1 + 0)) # encoding: [0xd3,0x01,0x00,0x00,0x00,0x00,0x00,0x12]
entry:
  %0 = load atomic i8, ptr %p acquire, align 1
  ret i8 %0
}

define dso_local i16 @load_acquire_i16(ptr nocapture noundef readonly %p) local_unnamed_addr {
; CHECK-LABEL: load_acquire_i16
; CHECK-LE: w0 = load_acquire((u16 *)(r1 + 0)) # encoding: [0xcb,0x10,0x00,0x00,0x12,0x00,0x00,0x00]
; CHECK-BE: w0 = load_acquire((u16 *)(r1 + 0)) # encoding: [0xcb,0x01,0x00,0x00,0x00,0x00,0x00,0x12]
entry:
  %0 = load atomic i16, ptr %p acquire, align 2
  ret i16 %0
}

define dso_local i32 @load_acquire_i32(ptr nocapture noundef readonly %p) local_unnamed_addr {
; CHECK-LABEL: load_acquire_i32
; CHECK-LE: w0 = load_acquire((u32 *)(r1 + 0)) # encoding: [0xc3,0x10,0x00,0x00,0x12,0x00,0x00,0x00]
; CHECK-BE: w0 = load_acquire((u32 *)(r1 + 0)) # encoding: [0xc3,0x01,0x00,0x00,0x00,0x00,0x00,0x12]
entry:
  %0 = load atomic i32, ptr %p acquire, align 4
  ret i32 %0
}

define dso_local i64 @load_acquire_i64(ptr nocapture noundef readonly %p) local_unnamed_addr {
; CHECK-LABEL: load_acquire_i64
; CHECK-LE: r0 = load_acquire((u64 *)(r1 + 0)) # encoding: [0xdb,0x10,0x00,0x00,0x12,0x00,0x00,0x00]
; CHECK-BE: r0 = load_acquire((u64 *)(r1 + 0)) # encoding: [0xdb,0x01,0x00,0x00,0x00,0x00,0x00,0x12]
entry:
  %0 = load atomic i64, ptr %p acquire, align 8
  ret i64 %0
}

define void @store_release_i8(ptr nocapture noundef writeonly %p,
                              i8 noundef signext %v) local_unnamed_addr {
; CHECK-LABEL: store_release_i8
; CHECK-LE: store_release((u8 *)(r1 + 0), w2) # encoding: [0xd3,0x21,0x00,0x00,0x23,0x00,0x00,0x00]
; CHECK-BE: store_release((u8 *)(r1 + 0), w2) # encoding: [0xd3,0x12,0x00,0x00,0x00,0x00,0x00,0x23]
entry:
  store atomic i8 %v, ptr %p release, align 1
  ret void
}

define void @store_release_i16(ptr nocapture noundef writeonly %p,
                               i16 noundef signext %v) local_unnamed_addr {
; CHECK-LABEL: store_release_i16
; CHECK-LE: store_release((u16 *)(r1 + 0), w2) # encoding: [0xcb,0x21,0x00,0x00,0x23,0x00,0x00,0x00]
; CHECK-BE: store_release((u16 *)(r1 + 0), w2) # encoding: [0xcb,0x12,0x00,0x00,0x00,0x00,0x00,0x23]
entry:
  store atomic i16 %v, ptr %p release, align 2
  ret void
}

define void @store_release_i32(ptr nocapture noundef writeonly %p,
                               i32 noundef %v) local_unnamed_addr {
; CHECK-LABEL: store_release_i32
; CHECK-LE: store_release((u32 *)(r1 + 0), w2) # encoding: [0xc3,0x21,0x00,0x00,0x23,0x00,0x00,0x00]
; CHECK-BE: store_release((u32 *)(r1 + 0), w2) # encoding: [0xc3,0x12,0x00,0x00,0x00,0x00,0x00,0x23]
entry:
  store atomic i32 %v, ptr %p release, align 4
  ret void
}

define void @store_release_i64(ptr nocapture noundef writeonly %p,
                               i64 noundef %v) local_unnamed_addr {
; CHECK-LABEL: store_release_i64
; CHECK-LE: store_release((u64 *)(r1 + 0), r2) # encoding: [0xdb,0x21,0x00,0x00,0x23,0x00,0x00,0x00]
; CHECK-BE: store_release((u64 *)(r1 + 0), r2) # encoding: [0xdb,0x12,0x00,0x00,0x00,0x00,0x00,0x23]
entry:
  store atomic i64 %v, ptr %p release, align 8
  ret void
}
