# RUN: not %lld -o /dev/null --irpgo-profile %s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# RUN: not %lld -o /dev/null --irpgo-profile=%s --call-graph-profile-sort 2>&1 | FileCheck %s --check-prefix=IRPGO-ERR
# IRPGO-ERR: --irpgo-profile is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --bp-compression-sort=function --call-graph-profile-sort %s 2>&1 | FileCheck %s --check-prefix=COMPRESSION-ERR
# COMPRESSION-ERR: --bp-compression-sort= is incompatible with --call-graph-profile-sort

# RUN: not %lld -o /dev/null --bp-compression-sort=malformed 2>&1 | FileCheck %s --check-prefix=COMPRESSION-MALFORM
# COMPRESSION-MALFORM: unknown value `malformed` for --bp-compression-sort=

# RUN: not %lld -o /dev/null --bp-compression-sort-startup-functions 2>&1 | FileCheck %s --check-prefix=STARTUP
# STARTUP: --bp-compression-sort-startup-functions must be used with --irpgo-profile
