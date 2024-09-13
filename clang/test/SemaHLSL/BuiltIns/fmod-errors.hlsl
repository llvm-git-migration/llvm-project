
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

float test_too_few_arg() {
  return __builtin_elementwise_fmod();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

float2 test_too_many_arg(float2 p0, float2 p1, float2 p3) {
  return __builtin_elementwise_fmod(p0, p1, p3);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float builtin_bool_to_float_type_promotion(bool p1, bool p2) {
  return __builtin_elementwise_fmod(p1, p2);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'bool')}}
}

float builtin_fmod_int_to_float_promotion(int p1, int p2) {
  return __builtin_elementwise_fmod(p1, p2);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}
}

float2 builtin_fmod_int2_to_float2_promotion(int2 p1, int2 p2) {
  return __builtin_elementwise_fmod(p1, p2);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int2' (aka 'vector<int, 2>'))}}
}

half builtin_fmod_double_type (double p0, double p1) {
  return __builtin_elementwise_fmod(p0, p1);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}

half builtin_fmod_double2_type (double2 p0, double2 p1) {
  return __builtin_elementwise_fmod(p0, p1);
  // expected-error@-1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values)}}
}
