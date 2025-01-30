// COM: Prior to the introduction of the FatLTO cleanup pass, this used to cause
// COM: the backend to crash, either due to an assertion failure, or because
// COM: the CFI instructions couldn't be correctly generated. So, check to make
// COM: sure that the FatLTO pipeline used by clang does not regress.

// COM: Check the generated IR doesn't contain llvm.type.checked.load in the final IR.
// RUN: %clang_cc1 -O1 -emit-llvm -o - -ffat-lto-objects \
// RUN:      -fvisibility=hidden \
// RUN:      -fno-rtti -fsanitize=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fsanitize-trap=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fwhole-program-vtables %s 2>&1 | FileCheck %s --check-prefix=FATLTO

// COM: Note that the embedded bitcode section will contain references to
// COM: llvm.type.checked.load, so we need to match the function body first.
// FATLTO-LABEL: entry:
// FATLTO-NEXT:   %vtable = load ptr, ptr %p1
// FATLTO-NOT: llvm.type.checked.load
// FATLTO-NEXT:   %vfunc = load ptr, ptr %vtable
// FATLTO-NEXT:   %call = tail call {{.*}} %vfunc(ptr {{.*}} %p1)
// FATLTO-NEXT:   ret void

// COM: Ensure that we don't crash in the backend anymore when clang uses
// COM: CFI checks with -ffat-lto-objects.
// RUN: %clang_cc1 -O1 --emit-codegen-only -ffat-lto-objects \
// RUN:      -fvisibility=hidden \
// RUN:      -fno-rtti -fsanitize=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fsanitize-trap=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fwhole-program-vtables %s

class a {
public:
  virtual long b();
};
void c(a &p1) { p1.b(); }
