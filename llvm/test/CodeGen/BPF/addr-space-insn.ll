; RUN: llc -march=bpfel -mcpu=v4 -filetype=asm -show-mc-encoding < %s | FileCheck %s

define dso_local void @test_fn(ptr addrspace(272) noundef %a, ptr addrspace(272) noundef %b) {
; CHECK-LABEL: test_fn:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    r2 = cast_kern(r2, 272)  # encoding: [0xbf,0x22,0x01,0x00,0x10,0x01,0x00,0x00]
; CHECK-NEXT:    r1 = cast_user(r1, 272)  # encoding: [0xbf,0x11,0x02,0x00,0x10,0x01,0x00,0x00]
; CHECK-NEXT:    *(u64 *)(r2 + 0) = r1
; CHECK-NEXT:    exit
entry:
  store volatile ptr addrspace(272) %a, ptr addrspace(272) %b, align 8
  ret void
}

declare ptr addrspace(272) @llvm.bpf.addr.space.p272.p272(ptr addrspace(272) nocapture, i32 immarg)
