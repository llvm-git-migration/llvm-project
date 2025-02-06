!RUN: %flang --target=aarch64-unknown-openbsd -### %s 2>&1 | FileCheck --check-prefixes=BACKTRACE %s

!BACKTRACE: -lexecinfo
