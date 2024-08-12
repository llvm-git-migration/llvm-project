! This only checks that the command line is correctly passed on to the
! -record-command-line option FC1 option and that the latter does not complain
! about anything. The correct lowering to a module attribute and beyond will
! be checked in other tests.
!
! RUN: %flang -### -frecord-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-RECORD %s
! RUN: %flang -### -fno-record-command-line %s 2>&1 | FileCheck --check-prefix=CHECK-NORECORD %s

! CHECK-RECORD: "-record-command-line" "{{.+}}/flang{{[^ ]*}} -### -frecord-command-line {{.+}}/frecord-command-line.f90 {{.*}}"

! CHECK-NORECORD-NOT: "-record-command-line"
