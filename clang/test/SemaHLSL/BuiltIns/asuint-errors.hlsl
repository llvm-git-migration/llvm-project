// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected


uint4 test_asuint_too_many_arg(float p0, float p1) {
  return asuint(p0, p1);
  // expected-error@-1 {{no matching function for call to 'asuint'}}
}

uint test_asuint_double(double p1) {
    return asuint(p1);
    // expected-error@-1{clang/20/include/hlsl/hlsl_details.h} {{static assertion failed due to requirement 'sizeof(unsigned int) == sizeof(double)'}}
}


uint test_asuint_half(half p1) {
    return asuint(p1);
    // expected-error@-1{clang/20/include/hlsl/hlsl_details.h} {{static assertion failed due to requirement 'sizeof(unsigned int) == sizeof(half)'}}
}
