// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=numerical | FileCheck -check-prefix=NSAN %s
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=numerical -fsanitize-ignorelist=%t | FileCheck -check-prefix=BL %s

// The sanitize_numerical_stability attribute should be attached to functions
// when NumericalStabilitySanitizer is enabled, unless no_sanitize_numerical_stability attribute
// is present.

// WITHOUT:  NoNSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoNSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// NSAN:  NoNSAN1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_numerical_stability))
int NoNSAN1(int *a) { return *a; }

// WITHOUT:  NoNSAN2{{.*}}) [[NOATTR]]
// BL:  NoNSAN2{{.*}}) [[NOATTR]]
// NSAN:  NoNSAN2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_numerical_stability))
int NoNSAN2(int *a);
int NoNSAN2(int *a) { return *a; }

// WITHOUT:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// NSAN:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize("numerical_stability")))
int NoNSAN3(int *a) { return *a; }

// WITHOUT:  NSANOk{{.*}}) [[NOATTR]]
// BL:  NSANOk{{.*}}) [[NOATTR]]
// NSAN: NSANOk{{.*}}) [[WITH:#[0-9]+]]
int NSANOk(int *a) { return *a; }

// WITHOUT:  TemplateNSANOk{{.*}}) [[NOATTR]]
// BL:  TemplateNSANOk{{.*}}) [[NOATTR]]
// NSAN: TemplateNSANOk{{.*}}) [[WITH]]
template<int i>
int TemplateNSANOk() { return i; }

// WITHOUT:  TemplateNoNSAN{{.*}}) [[NOATTR]]
// BL:  TemplateNoNSAN{{.*}}) [[NOATTR]]
// NSAN: TemplateNoNSAN{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize_numerical_stability))
int TemplateNoNSAN() { return i; }

int force_instance = TemplateNSANOk<42>()
                   + TemplateNoNSAN<42>();

