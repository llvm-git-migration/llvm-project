// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-MSBITFIELDS
// RUN: %clang -### -target x86_64-windows-gnu %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-MSBITFIELDS
// RUN: %clang -### -target x86_64-windows-msvc %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-MSBITFIELDS
// RUN: %clang -### -mms-bitfields %s 2>&1 | FileCheck %s -check-prefix=MSBITFIELDS
// RUN: %clang -### -mno-ms-bitfields %s 2>&1 | FileCheck %s -check-prefix=NO-MSBITFIELDS
// RUN: %clang -### -mno-ms-bitfields -mms-bitfields %s 2>&1 | FileCheck %s -check-prefix=MSBITFIELDS
// RUN: %clang -### -mms-bitfields -mno-ms-bitfields %s 2>&1 | FileCheck %s -check-prefix=NO-MSBITFIELDS

// MSBITFIELDS: -mms-bitfields
// NO-MSBITFIELDS: -mno-ms-bitfields
// DEFAULT-MSBITFIELDS-NOT: -mms-bitfields
// DEFAULT-MSBITFIELDS-NOT: -mno-ms-bitfields

