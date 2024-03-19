// RUN: %clang_cc1 -triple riscv64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

// expected-warning@+2 {{unsupported 'arch=rv64gcv' in the 'target_version' attribute string; 'target_version' attribute ignored}}
// expected-note@+1 {{previous definition is here}}
__attribute__((target_version("arch=rv64gcv"))) int fullArchString(void) { return 2; }
// expected-error@+2 {{redefinition of 'fullArchString'}}
// expected-warning@+1 {{unsupported 'arch=default' in the 'target_version' attribute string; 'target_version' attribute ignored}}
__attribute__((target_version("arch=default"))) int fullArchString(void) { return 2; }

// expected-warning@+2 {{unsupported 'mcpu=sifive-u74' in the 'target_version' attribute string; 'target_version' attribute ignored}}
// expected-note@+1 {{previous definition is here}}
__attribute__((target_version("mcpu=sifive-u74"))) int mcpu(void) { return 2; }
// expected-error@+1 {{redefinition of 'mcpu'}}
__attribute__((target_version("default"))) int mcpu(void) { return 2; }

// expected-warning@+2 {{unsupported 'mtune=sifive-u74' in the 'target_version' attribute string; 'target_version' attribute ignored}}
// expected-note@+1 {{previous definition is here}}
__attribute__((target_version("mtune=sifive-u74"))) int mtune(void) { return 2; }
// expected-error@+1 {{redefinition of 'mtune'}}
__attribute__((target_version("default"))) int mtune(void) { return 2; }

// expected-warning@+2 {{unsupported '' in the 'target_version' attribute string; 'target_version' attribute ignored}}
// expected-note@+1 {{previous definition is here}}
__attribute__((target_version(""))) int emptyVersion(void) { return 2; }
// expected-error@+1 {{redefinition of 'emptyVersion'}}
__attribute__((target_version("default"))) int emptyVersion(void) { return 2; }

// expected-note@+1 {{previous definition is here}}
__attribute__((target_version("arch=+c"))) int dupVersion(void) { return 2; }
// expected-error@+1 {{redefinition of 'dupVersion'}}
__attribute__((target_version("arch=+c"))) int dupVersion(void) { return 2; }
__attribute__((target_version("default"))) int dupVersion(void) { return 2; }
